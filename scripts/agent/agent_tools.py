"""
smolagents Tools for the ReAct agent -- 2 tool (+ final_answer nativo),
per indicazione dei tutor di limitare il numero di tool (più tool
confondevano misurabilmente questo modello da 3B nelle versioni precedenti):
- assess_retrieval_need: decide RETRIEVE vs ANSWER_DIRECTLY, sempre primo.
- retrieve_knowledge: recupera documenti INTERI (tutte le sezioni, non
  troncate) per i top-k candidati, poi lascia che ReAG-Critic selezioni
  quali singole SEZIONI sono rilevanti -- filtro fine sul testo completo,
  invece di troncare a monte sperando che la sezione utile sopravviva.
  Rispecchia l'esempio ufficiale di ReAG-Critic, che valuta sezioni
  individuali, non articoli interi.

refine_search è stato RIMOSSO in questo ridisegno: su tre design
indipendenti non ha mai alzato misurabilmente l'hit rate rispetto al solo
retrieve_knowledge, e i tutor hanno chiesto di limitare i tool a 2-3. Con
il recupero di documenti interi e il filtro fine per sezione, c'è già più
materiale utilizzabile per documento, riducendo il bisogno di una seconda
ricerca completa.

Tutta la generazione di testo che richiede giudizio (guesses, valutazione
del bisogno di retrieval) è delegata a chiamate Qwen dedicate, greedy, a
compito singolo -- non scritta dall'agente dentro il proprio JSON.

EpisodeState forza l'ordine previsto a livello di codice, non solo di
istruzione.
"""

from PIL import Image
from smolagents import Tool

GUESS_PROMPT_TEMPLATE = (
    "Look at this image and read the following question: '{question}'. "
    "Identify the specific entity (e.g., proper name, biological species, building) "
    "the question is asking about. Provide your top 3 most probable guesses for its identity. "
    "Output ONLY a comma-separated list of these 3 names "
    "(e.g., Fuchsia magellanica, Hibiscus rosa-sinensis, Mandevilla sanderi). "
    "Do not write full sentences, background descriptions, or explanations."
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


class EpisodeState:
    """Shared, per-episode state across both tools."""
    def __init__(self):
        self.has_assessed = False
        self.retrieval_recommended = None  # "RETRIEVE" | "ANSWER_DIRECTLY" | None
        self.has_retrieved = False
        self.last_labeled_sections = []    # lista filtrata di (label, testo)
        self.filter_removed_all = False    # True se il filtro ha scartato tutto
        self.n_sections_before_filter = 0
        self.n_sections_after_filter = 0

    def reset(self):
        self.has_assessed = False
        self.retrieval_recommended = None
        self.has_retrieved = False
        self.last_labeled_sections = []
        self.filter_removed_all = False
        self.n_sections_before_filter = 0
        self.n_sections_after_filter = 0


def _generate_dedicated(image: Image.Image, prompt_text: str, model_wrapper) -> str:
    """Chiamata dedicata, non-agentica, singola, greedy, instradata tramite
    generate_plain() del wrapper -- indipendente dal backbone."""
    return model_wrapper.generate_plain(image, prompt_text, max_new_tokens=40)


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

    def __init__(self, episode_state: EpisodeState, model_wrapper, **kwargs):
        super().__init__(**kwargs)
        self.episode_state = episode_state
        self.model_wrapper = model_wrapper
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
            raw = _generate_dedicated(self.current_image, prompt, self.model_wrapper)
        except Exception:
            raw = ""

        decision = "RETRIEVE"  # default conservativo se il parsing fallisce
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
        "Retrieves FULL Wikipedia articles (all sections) about the entity "
        "shown in the query image. Internally generates candidate identity "
        "guesses from the image and fuses them with your question to "
        "sharpen the search, then automatically selects only the individual "
        "sections judged relevant to your question before returning them. "
        "REQUIRES that assess_retrieval_need has been called first. Calling "
        "this again returns the exact same evidence."
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

    def __init__(self, retriever, episode_state: EpisodeState, model_wrapper, critic, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.episode_state = episode_state
        self.model_wrapper = model_wrapper
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
            rompt = GUESS_PROMPT_TEMPLATE.format(question=self.current_question)
            guesses = _generate_dedicated(self.current_image, prompt, self.model_wrapper)
        except Exception:
            guesses = ""

        combined_query = f"{guesses}"
        urls, labeled_sections = self.retriever.retrieve(
            self.current_image, query_text=combined_query, text_weight=0.3
        )
        self.retrieved_urls_log.extend(urls)
        self.episode_state.has_retrieved = True
        self.episode_state.n_sections_before_filter = len(labeled_sections)

        # Filtro fine per sezione: ReAG-Critic valuta OGNI sezione
        # individualmente (come nel suo esempio ufficiale), non l'intero
        # documento multi-sezione come unico blocco.
        filtered = self.critic.filter_passages(self.current_image, self.current_question, labeled_sections)
        self.episode_state.n_sections_after_filter = len(filtered)
        self.episode_state.filter_removed_all = bool(labeled_sections) and not filtered
        self.episode_state.last_labeled_sections = filtered

        if not filtered:
            if labeled_sections:
                return (
                    "Documents were retrieved, but a relevance filter judged "
                    "all sections irrelevant to the question. Answer from "
                    "the image and your own knowledge."
                )
            return "No relevant documents were found in the knowledge base for this image."

        return "\n\n".join(f"[{label}]\n{text}" for label, text in filtered)