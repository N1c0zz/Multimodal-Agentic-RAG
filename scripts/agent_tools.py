"""
smolagents Tools for the ReAct agent:
- KnowledgeRetrievalTool: image + auto-generated top-3 hypotheses (first pass)
- RefineSearchTool: image + agent-generated hypothesis (optional second pass)

KnowledgeRetrievalTool now internally generates its own top-3 hypotheses via
a dedicated (non-agentic, single-shot) Qwen call before retrieving -- the
same mechanism as the static "hypothesis-guided fusion" experiment (24.4%
aggregate, 25.9% hit rate). This is done automatically on every call, not
left to the agent's own tool-call arguments, so every episode benefits from
it rather than only the rare episodes where the agent chose to call
refine_search (32/1000 in an earlier run).

Both tools share a single RetrieverAgent instance (one EVA-CLIP load) and a
shared EpisodeState, which enforces that refine_search can only be called
AFTER retrieve_knowledge has run at least once in the current episode.

Each tool logs the URLs it retrieves during an episode (retrieved_urls_log),
so the main inference script can later check whether the oracle evidence was
retrieved at any point during the episode (for evidence_in_context logging).
"""

import torch
from PIL import Image
from smolagents import Tool
from retriever_agent import RetrieverAgent

GUESS_PROMPT = (
    "Analyze the main subject of this image. Provide your top 3 most probable "
    "guesses for its specific proper name, biological species, or exact "
    "identity. Output ONLY a comma-separated list of these 3 names "
    "(e.g., Fuchsia magellanica, Hibiscus rosa-sinensis, Mandevilla sanderi). "
    "Do not write full sentences, background descriptions, or explanations."
)


class EpisodeState:
    """Shared, per-episode state between the two retrieval tools."""
    def __init__(self):
        self.has_retrieved = False

    def reset(self):
        self.has_retrieved = False


class KnowledgeRetrievalTool(Tool):
    name = "retrieve_knowledge"
    description = (
        "Retrieves Wikipedia passages about the entity shown in the query image. "
        "Internally, this also generates candidate identity guesses from the "
        "image to sharpen the search. Use this FIRST, before answering, and "
        "before refine_search -- refine_search cannot be used until this has "
        "been called at least once. Calling this again will return the exact "
        "same evidence -- it will not give you new information. If the "
        "evidence is not enough, use refine_search instead, with your own "
        "hypothesis about the entity's identity."
    )
    inputs = {
        "reasoning": {
            "type": "string",
            "description": (
                "Briefly note what specific information you are looking for. "
                "This does not change the retrieved evidence, it is only for "
                "your own reasoning trace."
            ),
        }
    }
    output_type = "string"

    def __init__(self, retriever: RetrieverAgent, episode_state: EpisodeState, qwen_model, qwen_processor, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.episode_state = episode_state
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.current_image = None
        self.retrieved_urls_log = []

    def set_image(self, image: Image.Image):
        """Must be called once per episode/sample, before agent.run()."""
        self.current_image = image
        self.retrieved_urls_log = []

    def _generate_guesses(self, image: Image.Image) -> str:
        """
        Dedicated, non-agentic, single-shot Qwen call (greedy decoding) to
        produce up to 3 candidate identity guesses -- NOT part of the ReAct
        reasoning trace, does not consume an agent step.
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": GUESS_PROMPT},
            ],
        }]
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.qwen_processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.qwen_model.device)

        with torch.no_grad():
            gen_ids = self.qwen_model.generate(
                **inputs, max_new_tokens=40, do_sample=False,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        return self.qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def forward(self, reasoning: str) -> str:
        if self.current_image is None:
            return "Error: no query image is set for this episode."

        try:
            guesses = self._generate_guesses(self.current_image)
        except Exception:
            guesses = ""

        context, urls = self.retriever.retrieve(
            self.current_image, query_text=guesses, text_weight=0.3
        )
        self.retrieved_urls_log.extend(urls)
        self.episode_state.has_retrieved = True
        if not context:
            return "No relevant documents were found in the knowledge base for this image."
        return context


class RefineSearchTool(Tool):
    name = "refine_search"
    description = (
        "Refines the search using one or more of your OWN hypotheses about the "
        "entity's identity (e.g. candidate species or proper names), combined "
        "with the image. REQUIRES that retrieve_knowledge has already been "
        "called at least once in this episode -- calling this first will "
        "return an error. Use this only if the first evidence was insufficient "
        "or seemed to be about the wrong entity. Unlike calling "
        "retrieve_knowledge again, this CAN return different evidence."
    )
    inputs = {
        "hypothesis": {
            "type": "string",
            "description": (
                "One to three candidate identities for the entity shown in "
                "the image, separated by commas, based on the evidence "
                "already retrieved. Example: 'Fuchsia magellanica, Hibiscus "
                "rosa-sinensis'."
            ),
        }
    }
    output_type = "string"

    def __init__(self, retriever: RetrieverAgent, episode_state: EpisodeState, text_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.episode_state = episode_state
        self.text_weight = text_weight
        self.current_image = None
        self.retrieved_urls_log = []

    def set_image(self, image: Image.Image):
        """Must be called once per episode/sample, before agent.run()."""
        self.current_image = image
        self.retrieved_urls_log = []

    def forward(self, hypothesis: str) -> str:
        if self.current_image is None:
            return "Error: no query image is set for this episode."
        if not self.episode_state.has_retrieved:
            return (
                "Error: you must call retrieve_knowledge first, before using "
                "refine_search. Call retrieve_knowledge now."
            )
        context, urls = self.retriever.retrieve(
            self.current_image, query_text=hypothesis, text_weight=self.text_weight
        )
        self.retrieved_urls_log.extend(urls)
        if not context:
            return "No relevant documents were found for this hypothesis."
        return context