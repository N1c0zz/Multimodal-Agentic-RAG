"""
smolagents Tools for the ReAct agent (3 tools + built-in final_answer):
- assess_retrieval_need: decides RETRIEVE vs ANSWER_DIRECTLY, always first.
- retrieve_knowledge: image + auto-generated guesses + question (first pass).
- refine_search: agent decides whether to refine; a dedicated, grounded
  Qwen call re-guesses and searches again (optional second pass).

Relevance filtering via ReAG-Critic is now AUTOMATIC, applied inside
retrieve_knowledge and refine_search right after every retrieval, rather
than a separate optional filter_context tool. Rationale: in practice, the
agent called the equivalent optional tool in only ~1/5 sampled episodes,
including cases with obviously irrelevant context where it clearly should
have -- the same small-model reliability issue that already made us enforce
retrieve_knowledge itself outside the agent rather than trust the model to
remember. Filtering is cheap (a single forward pass per passage, no
generation), so there is no real cost to always applying it.

All judgment-requiring text generation (guesses, re-guesses, the retrieval
assessment) is delegated to dedicated, single-task, greedy Qwen calls, NOT
written by the agent inside its own tool-call JSON -- writing a query while
also producing tool-call syntax was found to measurably degrade quality.

EpisodeState enforces the intended step order programmatically (not just by
instruction), since prompt-level ordering was found unreliable at model scale.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from PIL import Image
from smolagents import Tool
from qwen_vl_utils import process_vision_info

from qwen_utils import generate_greedy
from retriever_agent import RetrieverAgent

GUESS_PROMPT = (
    "Analyze the main subject of this image. Provide your top 3 most probable "
    "guesses for its specific proper name, biological species, or exact "
    "identity. Output ONLY a comma-separated list of these 3 names "
    "(e.g., Fuchsia magellanica, Hibiscus rosa-sinensis, Mandevilla sanderi). "
    "Do not write full sentences, background descriptions, or explanations."
)

REFINE_GUESS_PROMPT = (
    "You are trying to identify the main subject of this image. A first "
    "retrieval attempt returned the following context, which may be about the "
    "WRONG entity:\n\n"
    "{context}\n\n"
    "Reconsider the image carefully. Provide your top 3 most probable guesses "
    "for its specific proper name, biological species, or exact identity, "
    "different from what the context above describes if that seems wrong. "
    "Output ONLY a comma-separated list of these 3 names. "
    "Do not write full sentences or explanations."
)

ASSESS_PROMPT_TEMPLATE = (
    "You are shown an image and a question about it. Decide whether you "
    "already know the answer with reasonable confidence using only your own "
    "knowledge and the image, or whether you need to look up external "
    "knowledge to answer reliably.\n\n"
    "Question: {question}\n\n"
    "Respond with EXACTLY one of these two words: RETRIEVE or ANSWER_DIRECTLY.\n"
    "Answer ANSWER_DIRECTLY if you can clearly identify the entity shown in "
    "the image and are reasonably confident about the answer from your own "
    "knowledge. Answer RETRIEVE if you are unsure of the entity's exact "
    "identity, or if the question asks for a specific fact (a number, date, "
    "or precise detail) that you cannot recall with confidence."
)

REFINE_CONTEXT_CHARS = 1200


class EpisodeState:
    """Shared, per-episode state across all three tools."""
    def __init__(self):
        self.has_assessed = False
        self.retrieval_recommended = None  # "RETRIEVE" | "ANSWER_DIRECTLY" | None
        self.has_retrieved = False
        self.first_context = ""            # filtered context from the most recent retrieval
        self.last_labeled_passages = []    # filtered list of (label, text)
        self.filter_removed_all = False    # True if the last filtering pass emptied all passages

    def reset(self):
        self.has_assessed = False
        self.retrieval_recommended = None
        self.has_retrieved = False
        self.first_context = ""
        self.last_labeled_passages = []
        self.filter_removed_all = False


def _generate_dedicated(image: Image.Image, prompt_text: str, qwen_model, qwen_processor) -> str:
    """
    Dedicated, non-agentic, single-shot, GREEDY Qwen call. Not part of the
    ReAct reasoning trace, does not consume an agent step. Used for guess
    generation and the retrieval-need assessment alike. Reuses
    qwen_utils.generate_greedy() for the actual generate+decode step, the
    same helper used by every non-agentic RAG script.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(qwen_model.device)

    return generate_greedy(qwen_model, qwen_processor, inputs, max_new_tokens=40)


def _filter_and_join(critic, image, question, labeled_passages):
    """
    Runs ReAG-Critic relevance filtering over labeled_passages and returns
    (filtered_passages, filtered_context_string). Shared by
    KnowledgeRetrievalTool and RefineSearchTool so both apply the exact same
    automatic filtering step after retrieval.
    """
    filtered = critic.filter_passages(image, question, labeled_passages)
    filtered_context = "\n\n".join(f"[{label}]\n{text}" for label, text in filtered)
    return filtered, filtered_context


class AssessRetrievalNeedTool(Tool):
    name = "assess_retrieval_need"
    description = (
        "Decides whether external knowledge retrieval is needed to answer "
        "this question, or whether you can answer directly from the image. "
        "MUST be called FIRST, before any other tool. Returns a recommendation: "
        "RETRIEVE or ANSWER_DIRECTLY."
    )
    inputs = {
        "reasoning": {
            "type": "string",
            "description": "Not used to change the assessment, only for your reasoning trace.",
        }
    }
    output_type = "string"

    def __init__(self, episode_state: EpisodeState, qwen_model, qwen_processor, **kwargs):
        super().__init__(**kwargs)
        self.episode_state = episode_state
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.current_image = None
        self.current_question = ""

    def set_image(self, image: Image.Image):
        self.current_image = image

    def set_question(self, question: str):
        self.current_question = question

    def forward(self, reasoning: str) -> str:
        if self.current_image is None:
            return "Error: no query image is set for this episode."

        prompt = ASSESS_PROMPT_TEMPLATE.format(question=self.current_question)
        try:
            raw = _generate_dedicated(self.current_image, prompt, self.qwen_model, self.qwen_processor)
        except Exception:
            raw = ""

        decision = "RETRIEVE"  # conservative default on parse failure
        if "ANSWER_DIRECTLY" in raw.upper() and "RETRIEVE" not in raw.upper():
            decision = "ANSWER_DIRECTLY"

        self.episode_state.has_assessed = True
        self.episode_state.retrieval_recommended = decision

        if decision == "RETRIEVE":
            return "Assessment: RETRIEVE. Call retrieve_knowledge next."
        return (
            "Assessment: ANSWER_DIRECTLY. You may answer directly using the "
            "image and your own knowledge with final_answer, or still call "
            "retrieve_knowledge if you want to double-check."
        )


class KnowledgeRetrievalTool(Tool):
    name = "retrieve_knowledge"
    description = (
        "Retrieves Wikipedia passages about the entity shown in the query "
        "image. Internally generates candidate identity guesses from the "
        "image and fuses them with your question to sharpen the search, and "
        "automatically filters out irrelevant passages before returning them. "
        "REQUIRES that assess_retrieval_need has been called first. Calling "
        "this again returns the exact same evidence. If the evidence seems to "
        "be about the wrong entity, use refine_search."
    )
    inputs = {
        "reasoning": {
            "type": "string",
            "description": (
                "Briefly note what you are looking for. This does not change "
                "the retrieved evidence, it is only for your reasoning trace."
            ),
        }
    }
    output_type = "string"

    def __init__(self, retriever: RetrieverAgent, episode_state: EpisodeState,
                 qwen_model, qwen_processor, critic, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.episode_state = episode_state
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.critic = critic
        self.current_image = None
        self.current_question = ""
        self.retrieved_urls_log = []

    def set_image(self, image: Image.Image):
        self.current_image = image
        self.retrieved_urls_log = []

    def set_question(self, question: str):
        self.current_question = question

    def forward(self, reasoning: str) -> str:
        if self.current_image is None:
            return "Error: no query image is set for this episode."
        if not self.episode_state.has_assessed:
            return (
                "Error: you must call assess_retrieval_need first, before "
                "using retrieve_knowledge. Call assess_retrieval_need now."
            )

        try:
            guesses = _generate_dedicated(
                self.current_image, GUESS_PROMPT, self.qwen_model, self.qwen_processor
            )
        except Exception:
            guesses = ""

        combined_query = f"Image tags: {guesses}. Question: {self.current_question}"
        _raw_context, urls, labeled_passages = self.retriever.retrieve(
            self.current_image, query_text=combined_query, text_weight=0.3
        )
        self.retrieved_urls_log.extend(urls)
        self.episode_state.has_retrieved = True

        # Automatic relevance filtering (see module docstring for why this
        # is no longer a separate, agent-discretionary tool).
        filtered_passages, filtered_context = _filter_and_join(
            self.critic, self.current_image, self.current_question, labeled_passages
        )
        self.episode_state.filter_removed_all = bool(labeled_passages) and not filtered_passages
        self.episode_state.last_labeled_passages = filtered_passages
        self.episode_state.first_context = filtered_context

        if not filtered_context:
            if labeled_passages:
                return (
                    "Documents were retrieved, but a relevance filter judged "
                    "all of them irrelevant to the question. Consider calling "
                    "refine_search, or answering from the image alone."
                )
            return "No relevant documents were found in the knowledge base for this image."
        return filtered_context


class RefineSearchTool(Tool):
    name = "refine_search"
    description = (
        "Requests a SECOND, different retrieval attempt when the first "
        "evidence seemed to be about the wrong entity. You do NOT need to "
        "provide a hypothesis: this tool reconsiders the image on its own, "
        "using the first evidence as a hint about what was wrong, searches "
        "again, and automatically filters out irrelevant passages. REQUIRES "
        "that retrieve_knowledge has already been called. Call this AT MOST "
        "ONCE, only if the first evidence looked incorrect."
    )
    inputs = {
        "reasoning": {
            "type": "string",
            "description": (
                "Briefly note why the first evidence seemed wrong. Only for "
                "your reasoning trace."
            ),
        }
    }
    output_type = "string"

    def __init__(self, retriever: RetrieverAgent, episode_state: EpisodeState,
                 qwen_model, qwen_processor, critic, text_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.episode_state = episode_state
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.critic = critic
        self.text_weight = text_weight
        self.current_image = None
        self.current_question = ""
        self.retrieved_urls_log = []

    def set_image(self, image: Image.Image):
        self.current_image = image
        self.retrieved_urls_log = []

    def set_question(self, question: str):
        self.current_question = question

    def forward(self, reasoning: str) -> str:
        if self.current_image is None:
            return "Error: no query image is set for this episode."
        if not self.episode_state.has_retrieved:
            return (
                "Error: you must call retrieve_knowledge first, before using "
                "refine_search. Call retrieve_knowledge now."
            )

        first_ctx = self.episode_state.first_context[:REFINE_CONTEXT_CHARS]
        prompt = REFINE_GUESS_PROMPT.format(context=first_ctx if first_ctx else "(no context retrieved)")
        try:
            guesses = _generate_dedicated(
                self.current_image, prompt, self.qwen_model, self.qwen_processor
            )
        except Exception:
            guesses = ""

        combined_query = f"Image tags: {guesses}. Question: {self.current_question}"
        _raw_context, urls, labeled_passages = self.retriever.retrieve(
            self.current_image, query_text=combined_query, text_weight=self.text_weight
        )
        self.retrieved_urls_log.extend(urls)

        # Automatic relevance filtering, same as retrieve_knowledge. Note
        # this ALSO covers the case seen in testing where refine_search
        # returns the exact same passages retrieve_knowledge already got
        # (image dominates the fused query at text_weight=0.3, so a
        # different textual hypothesis doesn't always shift the top-k) --
        # previously those already-rejected passages could silently be
        # reused by the agent without being re-filtered; now they always are.
        filtered_passages, filtered_context = _filter_and_join(
            self.critic, self.current_image, self.current_question, labeled_passages
        )
        self.episode_state.filter_removed_all = bool(labeled_passages) and not filtered_passages
        self.episode_state.last_labeled_passages = filtered_passages
        self.episode_state.first_context = filtered_context

        if not filtered_context:
            if labeled_passages:
                return (
                    "Documents were retrieved, but a relevance filter judged "
                    "all of them irrelevant to the question as well. Consider "
                    "answering from the image alone."
                )
            return "No relevant documents were found on the second attempt."
        return filtered_context