import os 

class Logger:
    def __init__(self, file_name, is_w = False) -> None:
        self.file_name = file_name
        self.is_write = is_w 

    def write(self, message):
        if self.is_write:
            message = message.strip('\n')
            message = message + '\n'
            self.file = open(self.file_name, 'a')
            self.file.write(message)
            print(message)
            self.file.close()
        else:
            print(message)
