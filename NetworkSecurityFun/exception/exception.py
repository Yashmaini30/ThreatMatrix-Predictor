import sys
from NetworkSecurityFun.logging import logger

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Custom exception class for handling and logging detailed error messages.
        """
        self.error_message = error_message
        _, _, exc_tb = error_detail.exc_info()
        
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error message: [{0}] at line number [{1}] in file name [{2}]".format(
            self.error_message, self.lineno, self.file_name
        )

if __name__ == "__main__":
    try:
        logger.logging.info("Logging has started")
        a = 1 / 0  # This will trigger ZeroDivisionError
        print("This will not be printed")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
