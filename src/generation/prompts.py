"""Fixed prompts for the thesis Llama 3 horror/thriller experiment."""

EN_PROMPT = (
    "Write a short literary scene in the horror/thriller genre.\n"
    "Focus on atmosphere, suspense, sensory details, and psychological tension.\n"
    "Do not explain the horror directly. Show a coherent scene."
)

RU_PROMPT = (
    "Напиши короткую литературную сцену в жанре ужасов/триллера.\n"
    "Сосредоточься на атмосфере, напряжении, сенсорных деталях и психологической тревоге.\n"
    "Не объясняй ужас напрямую. Покажи связную сцену."
)


def prompt_for_language(language: str) -> str:
    if language == "en":
        return EN_PROMPT
    if language == "ru":
        return RU_PROMPT
    raise ValueError(f"Unsupported language: {language}")
