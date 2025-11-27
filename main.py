# """
# WIKIPEDIA + OLLAMA CHATBOT (NO CHUNKING)
# - Uses gemma3:1b-instruct
# - LLM-based page ranking
# - Direct full-page Wikipedia extraction
# - Simplified architecture
# - Streaming answers
# """

# import re
# import wikipedia
# import ollama
# from spellchecker import SpellChecker


# class WikiChatbot:
#     def __init__(self, model="gemma3:1b"):
#         self.model = model
#         self.ollama = ollama.Client()
#         self.spell = SpellChecker()
#         wikipedia.set_lang("en")

#         print(f"✓ Chatbot initialized with model: {model}")

#     # -----------------------------------------------------
#     # TOPIC EXTRACTION (RULE-BASED)
#     # -----------------------------------------------------
#     def extract_topic(self, question):
#         q = question.lower()
#         remove_words = (
#             "what who why where when how explain define describe tell me about "
#             "characteristics features types information info about of"
#         ).split()

#         words = [w for w in re.findall(r"[a-zA-Z]+", q) if w not in remove_words]
#         if not words:
#             return question

#         topic = words[-1]
#         return self.spell.correction(topic)

#     # -----------------------------------------------------
#     # WIKIPEDIA SEARCH
#     # -----------------------------------------------------
#     def search(self, question, topic):
#         try:
#             r = wikipedia.search(question)
#         except:
#             r = []

#         if r:
#             return r

#         try:
#             r = wikipedia.search(topic)
#         except:
#             return []

#         return r

#     # -----------------------------------------------------
#     # LLM-BASED PAGE RANKING (NO CHUNKING)
#     # -----------------------------------------------------
#     def rank_pages(self, question, results):
#         """
#         Provide short summaries from up to 5 pages to LLM.
#         Ask LLM to pick the most relevant page.
#         """

#         snippets = []

#         for title in results[:5]:
#             try:
#                 page = wikipedia.page(title, auto_suggest=False)
#                 snippet = page.summary[:300]
#             except:
#                 snippet = "Unable to load summary."

#             snippets.append(f"### {title}\n{snippet}")

#         combined = "\n\n".join(snippets)

#         prompt = f"""
# Several Wikipedia pages were found. Choose the SINGLE BEST page
# that matches the user's question.

# User Question: "{question}"

# Pages:
# {combined}

# Return ONLY the exact page title. No explanation.
# """

#         resp = self.ollama.chat(
#             model=self.model,
#             messages=[{"role":"user","content":prompt}]
#         )

#         return resp["message"]["content"].strip()

#     # -----------------------------------------------------
#     # FETCH FULL PAGE CONTENT (NO CHUNKING)
#     # -----------------------------------------------------
#     def get_page(self, title):
#         try:
#             page = wikipedia.page(title, auto_suggest=True)
#             content = page.content

#             # Limit to avoid overwhelming 1B model
#             if len(content) > 6000:
#                 content = content[:6000] + "..."

#             return content
#         except:
#             return None

#     # -----------------------------------------------------
#     # STREAMED FINAL ANSWER
#     # -----------------------------------------------------
#     def stream_llm(self, prompt):
#         print("🤖 Generating answer...\n")
#         final = ""

#         for chunk in self.ollama.chat(
#             model=self.model,
#             messages=[{"role":"user","content":prompt}],
#             stream=True
#         ):
#             token = chunk["message"]["content"]
#             print(token, end="", flush=True)
#             final += token

#         return final

#     # -----------------------------------------------------
#     # MAIN ANSWER PIPELINE
#     # -----------------------------------------------------
#     def answer(self, question):
#         topic = self.extract_topic(question)
#         print(f"📚 Searching Wikipedia for: {topic}")

#         results = self.search(question, topic)

#         if not results:
#             msg = f"No Wikipedia pages found for '{topic}'."
#             print(msg)
#             return msg

#         print("🧠 Picking most relevant page using LLM...")
#         best_page = self.rank_pages(question, results)
#         print(f"📖 LLM selected: {best_page}")

#         content = self.get_page(best_page)
#         if not content:
#             msg = f"Could not fetch Wikipedia content for '{best_page}'."
#             print(msg)
#             return msg

#         # FINAL ANSWER PROMPT (No chunking)
#         prompt = f"""
# You are a helpful assistant. Use the Wikipedia content below to answer the question.

# Wikipedia Page: {best_page}

# Content:
# {content}

# User Question: "{question}"

# Provide a clear, structured answer using this format:
# - Definition
# - Key Background or History
# - Important Details
# - Additional Notes
# """

#         return self.stream_llm(prompt)


# # ---------------------------------------------------------
# # CLI LOOP
# # ---------------------------------------------------------
# def main():
#     print("=" * 60)
#     print("🤖 Wikipedia Chatbot (gemma3:1b-instruct, No Chunking)")
#     print("=" * 60)

#     bot = WikiChatbot()

#     while True:
#         q = input("\nYou: ").strip()

#         if q.lower() in ["exit", "quit", "bye"]:
#             print("👋 Goodbye!")
#             break

#         print()
#         bot.answer(q)
#         print("\n" + "-" * 60)


# if __name__ == "__main__":
#     main()
import re
import wikipedia
import ollama
from spellchecker import SpellChecker

wikipedia.set_lang("en")

class WikiChatbot:
    def __init__(self, model="gemma3:1b"):
        self.model = model
        self.ollama = ollama.Client()
        self.spell = SpellChecker()
        print(f"✓ Chatbot initialized with model: {model}")

    def extract_topic(self, question):
        """Extract the main topic from a question."""
        q = question.lower()
        remove_words = (
            "what who why where when how explain define describe tell me about "
            "characteristics features types information info about of"
        ).split()

        words = [w for w in re.findall(r"[a-zA-Z]+", q) if w not in remove_words]
        topic = self.spell.correction(words[-1]) if words else question
        return topic

    def search(self, question, topic):
        """Search Wikipedia for relevant pages."""
        try:
            r = wikipedia.search(question)
            if not r:
                r = wikipedia.search(topic)
        except Exception:
            r = []
        return r

    def rank_pages(self, question, results):
        """Rank search results and select the best page."""
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

        resp = self.ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        selected = resp["message"]["content"].strip()
        return selected

    def get_page(self, title):
        """Fetch Wikipedia page content."""
        try:
            page = wikipedia.page(title, auto_suggest=True)
            content = page.content
            if len(content) > 6000:
                content = content[:6000] + "..."
            return content
        except Exception:
            return None

    def stream_llm(self, prompt):
        """Stream LLM response token by token."""
        for chunk in self.ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            token = chunk["message"]["content"]
            yield token

    def answer_stream(self, question):
        """Main pipeline: search, rank, fetch, and generate answer."""
        try:
            # Extract topic
            topic = self.extract_topic(question)
            print(f"Extracted topic: {topic}")

            # Search Wikipedia
            results = self.search(question, topic)
            if not results:
                yield f"No Wikipedia pages found for '{topic}'."
                return
            print(f"Found {len(results)} results")

            # Rank pages
            best = self.rank_pages(question, results)
            print(f"Selected page: {best}")

            # Fetch page content
            content = self.get_page(best)
            if not content:
                yield f"Could not fetch page '{best}'."
                return

            # Prepare final prompt
            prompt = f"""
Use the following content to answer: {question}

{content}

Format:
- Definition
- Background
- Important Details
- Notes
"""

            # Stream final LLM response
            for token in self.stream_llm(prompt):
                yield token

        except Exception as e:
            yield f"\nError: {str(e)}"

    def answer(self, question):
        """Non-streaming version that returns complete answer."""
        result = ""
        for token in self.answer_stream(question):
            result += token
        return result


# Example usage
if __name__ == "__main__":
    bot = WikiChatbot()
    
    question = "What is machine learning?"
    print(f"\nQuestion: {question}\n")
    print("Answer:")
    
    # Streaming mode
    for token in bot.answer_stream(question):
        print(token, end="", flush=True)
    
    print("\n\n" + "="*50)
    
    # Non-streaming mode
    # answer = bot.answer("What is Python programming?")
    # print(answer)