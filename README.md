# SUDOKU SOLVER
## Project
This repository contains the code for a sudoku solving program. To keep the project simple, it has been decided to optimize the implementation for the www.sudoku.com page.

The program takes a screenshot of the web page with the sudoku, uses a computer vision model to extract the digits, solves the sudoku, and captures the keyboard and mouse to solve the sudoku on the web page.

<img src="img/sudoku.gif" width="800">

You can read an article explaining the implementation in detail in this [article](www.google.com).
## Dependencies
The Python libraries used are: Numpy, Pandas, Cv2, Pyautogui, Time, and Sklearn.

### Obtaining Sudoku
For this part of the program, the cv2 library is used to take the screenshot, process the image, detect the sudoku, and crop the screenshot containing it.
### Digit recognition
For this phase, a simple K-Nearest Neighbors (KNN) model is used, optimized to recognize the digits of the sudoku.com page.
### Sudoku resolution
Backtracking algorithms are used to solve the sudoku.
### Solving on the page
To solve the sudoku on the page, the Pyautogui library is used to perform the GUI Automation, once we know where on the screen the sudoku is located, and the solution of the sudoku.
