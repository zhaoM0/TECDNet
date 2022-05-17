import os 

class Logger:
    def __init__(self, file_name) -> None:
        self.file_name = file_name

    def write(self, message):
        message = message.strip('\n')
        message = message + '\n'
        self.file = open(self.file_name, 'a')
        self.file.write(message)
        print(message)
        self.file.close()

