"""
CODSOFT - Artificial Intelligence Internship
Task 1: Chatbot with Rule-Based Responses

A simple rule-based chatbot that matches user input against predefined
patterns (using regex/keyword matching) and returns an appropriate
response. No external NLP libraries are used.
"""

import re
import random


class RuleBasedChatbot:
    """
    A chatbot that identifies user intent using regex pattern matching
    and responds based on a predefined rule set.
    """

    def __init__(self, name="CodBot"):
        self.name = name

        # -----------------------------------------------------------
        # RULE SET
        # Each rule is a tuple: (intent_name, [regex_patterns], [responses])
        # The chatbot checks the user's input against each pattern list,
        # in order, and returns a random response from the first intent
        # that matches. This keeps rules easy to read and extend --
        # just add a new tuple to this list to teach the bot a new topic.
        # -----------------------------------------------------------
        self.rules = [
            (
                "greeting",
                [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"good (morning|afternoon|evening)"],
                [
                    f"Hello! I'm {self.name}. How can I help you today?",
                    "Hi there! What can I do for you?",
                ],
            ),
            (
                "farewell",
                [r"\bbye\b", r"goodbye", r"see you", r"\bexit\b", r"\bquit\b"],
                [
                    "Goodbye! Have a great day!",
                    "See you later! Feel free to come back anytime.",
                ],
            ),
            (
                "thanks",
                [r"\bthanks\b", r"\bthank you\b", r"appreciate it"],
                ["You're welcome!", "Anytime, happy to help!"],
            ),
            (
                "bot_identity",
                [r"who are you", r"what('?s| is) your name", r"about yourself"],
                [f"I'm {self.name}, a simple rule-based chatbot built for the CODSOFT AI internship."],
            ),
            (
                "wellbeing",
                [r"how are you", r"how('?s| is) it going"],
                ["I'm just a program, but I'm running smoothly! How about you?"],
            ),
            (
                "help",
                [r"\bhelp\b", r"what can you do", r"capabilities"],
                ["I can chat about greetings, farewells, my identity, and basic small talk. Try saying 'hi' or 'what can you do'."],
            ),
            (
                "time",
                [r"what time", r"current time"],
                ["I don't have access to a live clock right now, but you can check your device's clock!"],
            ),
            (
                "creator",
                [r"who (made|created|built) you", r"your creator"],
                ["I was built as part of the CODSOFT Artificial Intelligence internship, Task 1."],
            ),
        ]

        # Fallback responses used when no rule matches
        self.fallback_responses = [
            "Sorry, I didn't quite understand that. Could you rephrase?",
            "I'm not sure how to respond to that yet. Try asking something else!",
            "Hmm, that's beyond my current rule set. Could you try a different question?",
        ]

    def get_response(self, user_input: str) -> str:
        """
        Match user_input against the rule set and return a response.
        Matching is case-insensitive and uses substring/regex search,
        so the rule pattern doesn't need to match the whole input.
        """
        text = user_input.lower().strip()

        # Empty input edge case
        if not text:
            return "You didn't say anything -- I'm listening!"

        # Check each rule in order; return on first match
        for intent, patterns, responses in self.rules:
            for pattern in patterns:
                if re.search(pattern, text):
                    return random.choice(responses)

        # No rule matched -> fallback
        return random.choice(self.fallback_responses)

    def chat(self):
        """Run an interactive command-line chat loop."""
        print(f"{self.name}: Hi! Type 'bye' or 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            response = self.get_response(user_input)
            print(f"{self.name}: {response}")
            if re.search(r"\b(bye|goodbye|exit|quit)\b", user_input.lower()):
                break


# -----------------------------------------------------------------
# TEST RUN — demonstrates the chatbot handling several query types
# without needing manual input. Run this file directly to see it.
# -----------------------------------------------------------------
if __name__ == "__main__":
    bot = RuleBasedChatbot()

    test_inputs = [
        "Hello there!",                  # greeting
        "What can you do?",              # help
        "Who created you?",              # creator
        "asdkjhaslkdjh random text",     # unrecognized input -> fallback
        "Thanks a lot!",                 # thanks
        "Bye!",                          # farewell
    ]

    print("=== Automated Test Run ===\n")
    for msg in test_inputs:
        print(f"You: {msg}")
        print(f"{bot.name}: {bot.get_response(msg)}\n")

    # Uncomment the line below to chat interactively in the terminal:
    # bot.chat()
