import traceback
import sys

class CustomException(Exception):
    def __init__(self, errorMessage, errorDetail : sys):
        super().__init__(errorMessage)
        ## The super() function is a way to directly call a method from the parent 				
        ## class. So, super().__init__(errorMessage) is calling the __init__ method of the 			
        ## Exception class and passing your errorMessage to it.

        self.errorMessage = self.getDetailedErrorMessage(errorMessage, errorDetail)

    @staticmethod       ## We do not to create this class again and again to show our custom error messages
    ## A static method is essentially a function that logically belongs to a class but does not need 	## access to any instance-specific data. It doesn't use the self variable.
    def getDetailedErrorMessage(errorMessage, errorDetail : sys):
        
        _, _, exceptionTraceback = traceback.sys.exc_info()
        fileName = exceptionTraceback.tb_frame.f_code.co_filename
        lineNumber = exceptionTraceback.tb_lineno

        return f"Error in {fileName}, line {lineNumber} : {errorMessage}"
    
        ## The sys.exc_info() function is designed to return a tuple of three values about the exception that 	is currently being handled:
	## 1.The type of the exception (e.g., CustomException).
	## 2.The exception object itself.
	## 3.The traceback object, which contains all the information about where the error happened.
    
    def __str__(self):
        return self.errorMessage
