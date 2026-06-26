# Rule Based Chatbot - CodSoft AI Internship

def chatbot():
    print("Chatbot: Hello! How can I help you?")

    while True:
        user_input = input("You: ").lower()

        if user_input == "hello" or user_input == "hi":
            print("Chatbot: Hi! Nice to meet you.")

        elif "how are you" in user_input:
            print("Chatbot: I am good! Thanks for asking.")

        elif "your name" in user_input:
            print("Chatbot: I am a rule-based AI chatbot.")

        elif user_input == "bye":
            print("Chatbot: Goodbye! Have a nice day.")
            break

        else:
            print("Chatbot: Sorry, I don't understand that.")

chatbot()
