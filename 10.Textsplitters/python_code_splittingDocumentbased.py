from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Car:

    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start(self):
        print(f"{self.brand} {self.model} is starting.")

    def drive(self):
        print(f"{self.brand} {self.model} is driving.")

    def stop(self):
        print(f"{self.brand} {self.model} has stopped.")


# Object creation
car1 = Car("Toyota", "Camry")

# Calling methods
car1.start()
car1.drive()
car1.stop()

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)

result = splitter.split_text(text)
print(len(result))
print(result[0])
print("**************************")
print(result[1])