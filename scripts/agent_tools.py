"""
smolagents Tools for the ReAct agent:
- KnowledgeRetrievalTool: image-only retrieval (first pass)
- RefineSearchTool: image + agent-generated hypothesis fused retrieval (second pass)

Both share a single RetrieverAgent instance (one EVA-CLIP load) and a shared
EpisodeState, which enforces that refine_search can only be called AFTER
retrieve_knowledge has run at least once in the current episode. Without this
guardrail, the model sometimes skips straight to refine_search with an
ungrounded (guessed) hypothesis, which can bias the fused query away from the
correct entity -- worse than a plain image-only search.

Each tool logs the URLs it retrieves during an episode (retrieved_urls_log),
so the main inference script can later check whether the oracle evidence was
retrieved at any point during the episode (for evidence_in_context logging).
"""

from PIL import Image
from smolagents import Tool
from retriever_agent import RetrieverAgent


class EpisodeState:
    """Shared, per-episode state between the two retrieval tools."""
    def __init__(self):
        self.has_retrieved = False

    def reset(self):
        self.has_retrieved = False


class KnowledgeRetrievalTool(Tool):
    name = "retrieve_knowledge"
    description = (
        "Retrieves Wikipedia passages about the entity shown in the query image, "
        "found via image similarity search over a knowledge base. Use this FIRST, "
        "before answering, and before refine_search -- refine_search cannot be "
        "used until this has been called at least once. Because this retrieval "
        "is based only on the image, calling it again will return the exact "
        "same evidence -- it will not give you new information. If the evidence "
        "is not enough, use refine_search instead, with a specific hypothesis "
        "about the entity's identity."
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

    def __init__(self, retriever: RetrieverAgent, episode_state: EpisodeState, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.episode_state = episode_state
        self.current_image = None
        self.retrieved_urls_log = []

    def set_image(self, image: Image.Image):
        """Must be called once per episode/sample, before agent.run()."""
        self.current_image = image
        self.retrieved_urls_log = []

    def forward(self, reasoning: str) -> str:
        if self.current_image is None:
            return "Error: no query image is set for this episode."
        context, urls = self.retriever.retrieve(self.current_image, query_text="", text_weight=0.0)
        self.retrieved_urls_log.extend(urls)
        self.episode_state.has_retrieved = True
        if not context:
            return "No relevant documents were found in the knowledge base for this image."
        return context


class RefineSearchTool(Tool):
    name = "refine_search"
    description = (
        "Refines the search using a specific hypothesis about the entity's identity "
        "(e.g. a candidate species or proper name), combined with the image. "
        "REQUIRES that retrieve_knowledge has already been called at least once in "
        "this episode -- calling this first will return an error. Use this only if "
        "the first evidence was insufficient or seemed to be about the wrong entity, "
        "and you have a more specific guess based on what you already read. Unlike "
        "calling retrieve_knowledge again, this CAN return different evidence."
    )
    inputs = {
        "hypothesis": {
            "type": "string",
            "description": (
                "Your best guess at the specific identity of the entity shown in the "
                "image (e.g. a species name, proper name, or category), based on the "
                "evidence already retrieved. This text is combined with the image to "
                "refine the search."
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