# Task 1: Chatbot with Rule-Based Responses

## CODSOFT Artificial Intelligence Internship

###  Overview
This project is a simple **rule-based chatbot** built in Python. It identifies
user intent by matching input text against predefined regex/keyword patterns
and responds using a structured set of rules — no external NLP libraries required.

This task demonstrates the fundamentals of:
- Natural language pattern matching
- Conditional/rule-based logic
- Basic conversational flow design

---

###  How It Works
1. The user's input is lowercased and cleaned.
2. The chatbot checks the input against a list of **rules**, where each rule
   contains:
   - an **intent name** (e.g. `greeting`, `farewell`)
   - a list of **regex patterns** used to detect that intent
   - a list of possible **responses**
3. The first rule whose pattern matches the input is used, and a response
   is chosen (randomly, for variety).
4. If no rule matches, a **fallback response** is returned, prompting the
   user to rephrase.

This design makes it easy to add new topics — just append a new rule
tuple to the `self.rules` list in `chatbot.py`.

---

###  Supported Intents
| Intent         | Example Trigger Phrases                     |
|----------------|----------------------------------------------|
| Greeting       | "hi", "hello", "good morning"                |
| Farewell       | "bye", "goodbye", "quit", "exit"             |
| Thanks         | "thanks", "thank you"                        |
| Bot Identity   | "who are you", "what's your name"            |
| Wellbeing      | "how are you"                                |
| Help           | "help", "what can you do"                    |
| Time           | "what time is it"                            |
| Creator        | "who made you", "your creator"               |
| *Unrecognized* | anything not matching the above → fallback   |

---

###  How to Run

**Requirements:** Python 3.x (no external libraries needed)

```bash
python chatbot.py
```

By default, running the file executes an **automated test run** showing
the chatbot responding to 6 different sample inputs (greeting, help,
creator, unrecognized input, thanks, farewell).

To chat with the bot **interactively** in your terminal, open `chatbot.py`
and uncomment the last line:

```python
bot.chat()
```

Then run the script again and type your messages. Type `bye`, `quit`,
or `exit` to end the conversation.

---

###  Files
```
chatbot.py    # Full chatbot implementation + test run
README.md     # Project documentation (this file)
```

---

###  Example Interaction
```
You: Hello there!
CodBot: Hi there! What can I do for you?

You: What can you do?
CodBot: I can chat about greetings, farewells, my identity, and basic small talk...

You: asdkjhaslkdjh random text
CodBot: Sorry, I didn't quite understand that. Could you rephrase?

You: Bye!
CodBot: Goodbye! Have a great day!
```

---

###  Possible Extensions
- Add more intents/rules (weather, jokes, FAQs)
- Use `difflib` or fuzzy matching for typo tolerance
- Wrap in a simple web UI (Flask/Streamlit)
- Add memory of conversation context for follow-up questions

---

### 🏷️ Internship Info
This task was completed as part of the **CODSOFT Artificial Intelligence
Internship**. Repository maintained under: `CODSOFT`

#codsoft #internship #artificialintelligence
