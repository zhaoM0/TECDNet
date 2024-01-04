import os 
import datetime

class Logger:
    def __init__(self, file_name, write_log_file=False) -> None:
        self.file_name = file_name
        self.is_write_log_file = write_log_file
                
        mode = 'w' if not os.path.exists(self.file_name) else 'a'
        self.file = open(self.file_name, mode)
       
        if mode == 'w':
            self.file.write("TECDNet Training Log File\n")
            self.file.write("--------------------------------------------\n")
            self.file.flush()

    def write(self, message):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = message.strip('\n') + '\n'
        
        if self.is_write_log_file:
            with open(self.file_name, 'a') as f:
                f.write(f"{current_time}: {message}")
                
        print(message)
