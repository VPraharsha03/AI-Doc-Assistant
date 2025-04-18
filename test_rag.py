from chat_ui import query_rag
from models import Models

TEST_EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_yeti_1():
    assert query_and_validate(
        question="What is the most well-known species of Yeti?",
        expected_response="The most well-known species is the Classic Yeti, also known as the Abominable Snowman.",
    )

def test_yeti_2():
    assert query_and_validate(
        question="What are the physical characteristics of the Classic Yeti?",
        expected_response="The Classic Yeti is described as a large, ape-like creature covered in white or gray fur, standing between 6 to 10 feet tall.",
    )

def test_yeti_3():
    assert query_and_validate(
        question="Where is the Classic Yeti primarily reported to inhabit?",
        expected_response="The Classic Yeti is primarily reported to inhabit the snow-covered mountains and forests of the Himalayas, particularly in Nepal and Tibet.",
    )

def test_yeti_4():
    assert query_and_validate(
        question="What is the Red Yeti also known as?",
        expected_response="The Red Yeti is also known as Mete.",
    )

def test_yeti_5():
    assert query_and_validate(
        question="How does the Red Yeti differ from the Classic Yeti?",
        expected_response="The Red Yeti has reddish-brown fur and a more human-like appearance, and it inhabits lower forested regions.",
    )

def test_yeti_6():
    assert query_and_validate(
        question="What is the cultural significance of the Red Yeti?",
        expected_response="The Red Yeti is considered a protector of the forest and is associated with local tribes.",
    )

def test_yeti_7():
    assert query_and_validate(
        question="What is the Black Yeti also referred to as?",
        expected_response="The Black Yeti is referred to as Maha.",
    )

def test_yeti_8():
    assert query_and_validate(
        question="What does the term Himalayan Yeti refer to?",
        expected_response="The term Himalayan Yeti is a generalized term that encompasses various sightings and reports of Yeti-like creatures across the entire Himalayan range, often exhibiting features of both the Classic and Red Yetis.",
    )

def test_yeti_9():
    assert query_and_validate(
        question="What are the characteristics of the Black Yeti?",
        expected_response="The Black Yeti is characterized by its dark fur and is often described as being more aggressive than other Yeti species.",
    )

def test_yeti_10():
    assert query_and_validate(
        question="What is the habitat of the Black Yeti?",
        expected_response="The Black Yeti is said to inhabit rocky terrains and caves, making it less visible to humans.",
    )

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = TEST_EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    models = Models()
    model = models.chat_model
    eval_results_str = model.invoke(prompt)

    print(prompt)

    if "true" in eval_results_str.content:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {eval_results_str}" + "\033[0m")
        return True
    elif "false" in eval_results_str.content:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {eval_results_str}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
