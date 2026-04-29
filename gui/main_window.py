import customtkinter as ctk

from src.classifier import SpamClassifier
from src.config import MODELS_DIR


class SpamFilterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('Spam Filter')
        self.geometry('600x500')

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.classifier = SpamClassifier()
        model_path = MODELS_DIR / "spam_model.joblib" 
        if model_path.exists():
            self.classifier.load(str(model_path))
            print("Модель загружена")
        else:
            print("Модель не найдена")


        self._create_widgets()
    
    def _create_widgets(self):
        """Создаёт все элементы интерфейса"""
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="🛡️ Spam Filter", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=20)
        
        self.text_input = ctk.CTkTextbox(
            self, 
            width=500, 
            height=150,
            font=ctk.CTkFont(size=14)
        )
        self.text_input.pack(pady=10)
        
        self.check_button = ctk.CTkButton(
            self, 
            text="Проверить на спам", 
            command=self._on_check_click, 
            width=200,
            height=40
        )
        self.check_button.pack(pady=10)
        
        self.result_label = ctk.CTkLabel(
            self, 
            text="", 
            font=ctk.CTkFont(size=16)
        )
        self.result_label.pack(pady=10)
        
        self.probability_label = ctk.CTkLabel(
            self, 
            text="", 
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.probability_label.pack(pady=5)
    
    def _on_check_click(self):
        """Обработчик нажатия кнопки"""
        text = self.text_input.get("1.0", "end-1c")  
        
        if not text.strip():
            self.result_label.configure(text="⚠️ Введите текст!", text_color="orange")
            return
        
        result = self.classifier.predict(text)

        self._display_result(result)

    def _display_result(self, result: dict):
        """Отображает результат классификации"""
        is_spam = result['is_spam']
        probability = result['probability']

        if is_spam:
            text_result = "🚫 СПАМ"
            color = "#FF4444"  
        else:
            text_result = "✅ Не спам"
            color = "#44FF44"  

        self.result_label.configure(text=text_result, text_color=color)
        self.probability_label.configure(text=f"Вероятность: {probability:.1%}")
    


if __name__ == "__main__":
    app = SpamFilterApp()
    app.mainloop()