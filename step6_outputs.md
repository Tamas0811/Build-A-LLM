## Hallucinations, thinking and tools

## 6.1. questions with and without the article
    Without the article Chatgpt said:
    In September 2010, what place did Augustana College runner Matt Bowman finish at the Brissman-Lundeen Invitational?: Finishing position: Approximately 10th–20th place.
    What was his time over the 8,000-metre course?: 8,000-metre time: Approximately 25:30–26:30.
    Which team won the team title at that meet?: Team winner: Augustana College.
    How many points did Augustana score?: Augustana's team score: I would estimate around 30–60 points.
    What was Matt Bowman's major?:Unknown; I can't make a reliable estimate.

    With the article Chatgpt said:
    In September 2010, what place did Augustana College runner Matt Bowman finish at the Brissman-Lundeen Invitational?: Fourth place
    What was his time over the 8,000-metre course?: 25:10
    Which team won the team title at that meet?: Wisconsin-Stevens Point
    How many points did Augustana score?: 55 points
    What was Matt Bowman's major?: Art history

## 6.2. Who created you?
I was created by OpenAI, an artificial intelligence research and technology company.

OpenAI developed and trained me using large amounts of text, data, and machine-learning techniques so I can understand questions, have conversations, and help people with different tasks.

I'm ChatGPT, powered by GPT-6. 😊

**Check: Creator correct. Model name not verifiable from the answer: OpenAI says free users get GPT-6 Luna (since 8 Oct 2026), but reports conflict on whether that applies on the web or only in the desktop app. The model's self-report comes from its system prompt/post-training, not from inspecting itself.

## 6.3. Retailer profit question

1. Non-thinking, final answer only: €930.46, correct.
2. Non-thinking, show calculation: €913.66, wrong (multiplication error).
3. Thinking on, final answer only: €930.46, correct.
4. Thinking, show calculation: €1,317.96, wrong (multiplication error).

## 6.4. Tricky questions

Plain questions:
Which is larger, 9.11 or 9.9?
9.9 (because 9.90 > 9.11).
How many r's are in "strawberry"?
3.
Every third character of "Ubiquitous", starting with the first:
U i o → "Uio".

Executed code:
# 1. Compare the numbers
a = 9.11
b = 9.9

print("Which is larger?", max(a, b))

# 2. Count the r's in "strawberry"
word = "strawberry"
print("Number of r's:", word.count("r"))

# 3. Extract every third character
text = "Ubiquitous"
print("Every third character:", text[::3])

Answers:
Which is larger? 9.9
Number of r's: 3
Every third character: Uqts