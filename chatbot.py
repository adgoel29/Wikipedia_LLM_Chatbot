import re
import wikipedia
import ollama
from spellchecker import SpellChecker
from langfuse import Langfuse
from dotenv import load_dotenv
import os
import traceback

load_dotenv()
wikipedia.set_lang("en")

class WikiChatbot:
    def __init__(self, model="gemma3:1b"):
        self.model = model
        self.ollama = ollama.Client()
        self.spell = SpellChecker()
        self.lf = Langfuse()  # uses env variables
        print(f"✓ Chatbot initialized with model: {model}")

    # -----------------------------------------------------
    def extract_topic(self, question, parent_trace=None):
        trace = self.lf.trace(name="extract_topic", input={"question": question})
        try:
            q = question.lower()
            remove_words = (
                "what who why where when how explain define describe tell me about "
                "characteristics features types information info about of"
            ).split()

            words = [w for w in re.findall(r"[a-zA-Z]+", q) if w not in remove_words]
            topic = self.spell.correction(words[-1]) if words else question

            trace.output = {"topic": topic}
            return topic
        except Exception as e:
            trace.output = {"error": str(e)}
            raise
        # finally:
            # trace.end()

    # -----------------------------------------------------
    def search(self, question, topic):
        trace = self.lf.trace(name="wikipedia_search", input={"question": question, "topic": topic})
        try:
            try:
                r = wikipedia.search(question)
                if not r:
                    r = wikipedia.search(topic)
            except Exception:
                r = []
            trace.output = {"results": r}
            return r
        except Exception as e:
            trace.output = {"error": str(e)}
            raise
        # finally:
            # trace.end()

    # -----------------------------------------------------
    def rank_pages(self, question, results):
        trace = self.lf.trace(name="rank_pages", input={"question": question, "results": results})
        span = None
        try:
            snippets = []
            for title in results[:5]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    snippet = page.summary[:300]
                except Exception:
                    snippet = "Unable to load summary."

                snippets.append(f"### {title}\n{snippet}")

            combined = "\n\n".join(snippets)

            prompt = f"""
Several Wikipedia pages were found. Choose the SINGLE BEST page.

User Question: "{question}"

Pages:
{combined}

Return ONLY the page title.
"""

            # create an explicit span for the LLM rank call (linked to the trace)
            span = self.lf.span(trace_id=trace.id, name="ollama_rank_call", input={"prompt": prompt})
            resp = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            selected = resp["message"]["content"].strip()

            span.output = {"selected": selected}
            span.end()

            trace.output = {"best_page": selected}
            return selected
        except Exception as e:
            if span:
                try:
                    span.output = {"error": str(e)}
                    span.end()
                except Exception:
                    pass
            trace.output = {"error": str(e)}
            raise
        # finally:
        #     trace.end()

    # -----------------------------------------------------
    def get_page(self, title):
        trace = self.lf.trace(name="fetch_page", input={"title": title})
        try:
            try:
                page = wikipedia.page(title, auto_suggest=True)
                content = page.content
                if len(content) > 6000:
                    content = content[:6000] + "..."
            except Exception:
                content = None

            trace.output = {"content_found": bool(content)}
            return content
        except Exception as e:
            trace.output = {"error": str(e)}
            raise
        # finally:
        #     trace.end()

    # -----------------------------------------------------
    def stream_llm(self, prompt, parent_trace_id=None):
        # create a span specifically for the streaming LLM call (linked to parent trace)
        span = None
        trace = None
        try:
            if parent_trace_id:
                span = self.lf.span(trace_id=parent_trace_id, name="ollama_stream_call", input={"prompt": prompt})
            else:
                # fallback trace if no parent
                trace = self.lf.trace(name="final_answer_generation", input={"prompt": prompt})
                span = self.lf.span(trace_id=trace.id, name="ollama_stream_call", input={"prompt": prompt})

            # synchronous streaming from ollama; yield tokens to caller
            for chunk in self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ):
                token = chunk["message"]["content"]
                # we can emit partial token info in span output if desired (lightweight)
                yield token

            span.output = {"status": "completed"}
            span.end()
            if trace:
                trace.output = {"status": "completed"}
                # trace.end()
        except Exception as e:
            # ensure span/trace are closed with error info
            try:
                if span:
                    span.output = {"error": str(e)}
                    span.end()
            except Exception:
                pass
            try:
                if trace:
                    trace.output = {"error": str(e)}
                    # trace.end()
            except Exception:
                pass
            raise

    # -----------------------------------------------------
    def answer_stream(self, question):
        outer = self.lf.trace(name="answer_pipeline", input={"question": question})
        final_answer = ""   # ⬅ WE WILL STORE FULL ANSWER HERE

        try:
            # extract topic
            try:
                topic = self.extract_topic(question)
            except Exception as e:
                msg = f"Error extracting topic: {str(e)}"
                outer.output = {"error": msg}
                # outer.finish()
                yield msg
                return

            # search
            try:
                results = self.search(question, topic)
                if not results:
                    msg = f"No Wikipedia pages found for '{topic}'."
                    outer.output = {"error": msg}
                    # outer.finish()
                    yield msg
                    return
            except Exception as e:
                msg = f"Error during search: {str(e)}"
                outer.output = {"error": msg}
                # outer.finish()
                yield msg
                return

            # rank pages
            try:
                best = self.rank_pages(question, results)
            except Exception as e:
                msg = f"Error selecting page: {str(e)}"
                outer.output = {"error": msg}
                # outer.finish()
                yield msg
                return

            # fetch page
            try:
                content = self.get_page(best)
                if not content:
                    msg = f"Could not fetch page '{best}'."
                    outer.output = {"error": msg}
                    # outer.finish()
                    yield msg
                    return
            except Exception as e:
                msg = f"Error fetching page: {str(e)}"
                outer.output = {"error": msg}
                # outer.finish()
                yield msg
                return

            # prepare final prompt
            prompt = f"""
    Use the following content to answer:

    {content}

    Format:
    - Definition
    - Background
    - Important Details
    - Notes
    """

            # ────────────────────────────────────────────────
            # STREAM FINAL LLM RESPONSE
            # ────────────────────────────────────────────────
            try:
                for token in self.stream_llm(prompt, parent_trace_id=outer.id):
                    final_answer += token       # ⬅ APPEND TOKEN
                    yield token                 # ⬅ RETURN TOKEN TO FRONTEND
            except Exception as e:
                msg = f"Error generating answer: {str(e)}"
                outer.output = {"error": msg}
                # outer.finish()
                yield "\n" + msg
                return

            # ────────────────────────────────────────────────
            # STORE FINAL ANSWER IN LANGFUSE
            # ────────────────────────────────────────────────
            outer.output = {
                "status": "completed",
                "page_used": best,
                "final_answer": final_answer,   # ⬅ NOW LANGFUSE CAN SEE FULL RESPONSE
            }
            # outer.finish()

        except Exception as e_outer:
            outer.output = {"error": str(e_outer)}
            # outer.finish()
            yield f"\nFatal error: {str(e_outer)}"
            
