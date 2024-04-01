##1d array sum, searching, min max even odd
##Sum of Elements in an Array
def sum_of_elements(arr):
    total_sum = sum(arr)  # Using built-in sum function
    return total_sum
arr = [1, 2, 3, 4, 5]
print("Sum of elements:", sum_of_elements(arr))
# Time Complexity: O(n), where n is the number of elements in the array.

##Searching an Element in an Array
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
target = 3
print("Element found at index:", linear_search(arr, target))
# Time Complexity: O(n), because in the worst case, we might have to check each element.

##Finding Minimum and Maximum Element in an Array
def find_min_max(arr):
    minimum = min(arr)
    maximum = max(arr)
    return minimum, maximum
print("Minimum and Maximum:", find_min_max(arr))
# Time Complexity: O(n) for each, min and max functions. So, effectively O(n).

##Count the Number of Even and Odd Numbers in an Array
def count_even_odd(arr):
    even_count = sum(1 for x in arr if x % 2 == 0)
    odd_count = sum(1 for x in arr if x % 2 != 0)
    return even_count, odd_count
print("Even and Odd counts:", count_even_odd(arr))
# Time Complexity: O(n), as we go through each element once to check if it's even or odd.

##Comparison of Methods
##Sum of Elements: The direct use of Python's built-in sum function is the most efficient and straightforward way to calculate the sum of elements, with a time complexity of O(n).
##Searching an Element: We demonstrated linear search, which is O(n). For unsorted arrays, this is the best we can do without additional information. If the array is sorted, binary search could be used instead, lowering the time complexity to O(log n).
##Finding Minimum and Maximum Elements: Using Python's built-in min and max functions is very efficient. If both minimum and maximum are needed at once, doing a single pass through the array could be slightly more efficient in practice, as it would still be O(n) but with half the number of total comparisons.
##Counting Even and Odd Numbers: Our approach using list comprehensions (or generator expressions) is concise and runs in O(n). This is as efficient as it gets for this problem, given that every element needs to be inspected.





##2d array row, col, diag elmnt sum & sum multi of 2 matrix

##Row Sum of a Matrix
def row_sum(matrix):
    return [sum(row) for row in matrix]
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(row_sum(matrix))
##Time Complexity: O(n^2), where n is the number of rows We iterate over each element of the matrix once.


##Column Sum of a Matrix
def column_sum(matrix):
    return [sum(row[i] for row in matrix) for i in range(len(matrix))]
print(column_sum(matrix))
##Time Complexity: O(n^2), We iterate over every element once, but in a different order.


##Sum of Diagonal Elements
def diagonal_sum(matrix):
    n = len(matrix)
    primary_diag = sum(matrix[i][i] for i in range(n))
    secondary_diag = sum(matrix[i][n-i-1] for i in range(n))
    return primary_diag, secondary_diag
print(diagonal_sum(matrix))
##Time Complexity: O(n), where n is the size of the matrix. We only iterate through each diagonal once.


##Addition of Two Matrices
def add_matrices(matrix1, matrix2):
    return [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
matrix2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
print(add_matrices(matrix, matrix2))
##Time Complexity: O(n^2), since we need to add every corresponding element of the two matrices.


##Multiplication of Two Matrices
def multiply_matrices(matrix1, matrix2):
    n = len(matrix1)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result
print(multiply_matrices(matrix, matrix2))
##Time Complexity: O(n^3), due to the triple nested loop where we compute the sum of products for each element of the result matrix.





##list based stack operation
class ListStack:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
    def pop(self):
        """Remove and return the top item of the stack."""
        if self.isEmpty():
            return "Stack is empty"
        return self.items.pop()
    def peek(self):
        """Return the top item of the stack without removing it."""
        if self.isEmpty():
            return "Stack is empty"
        return self.items[-1]
    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)
    def __str__(self):
        """Return a string representation of the stack."""
        return str(self.items)
if __name__ == "__main__":
    stack = ListStack()
    print("Is the stack empty?", stack.isEmpty())
    stack.push(4)
    stack.push('apple')
    stack.push(10)
    print("Current stack:", stack)
    print("Stack size:", stack.size())
    print("Top item:", stack.peek())
    stack.pop()
    print("Stack after popping:", stack)
    print("Is the stack empty?", stack.isEmpty())





##linear and binary search
import time
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1
def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
    
    while low <= high:
        mid = (high + low) // 2
        
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1
arr = sorted([i for i in range(1, 10001)])  
x = 9999
start_time = time.time()
result = linear_search(arr, x)
end_time = time.time()
print(f"Linear Search: Found at index {result}. Time taken: {(end_time - start_time) * 1000:.6f} ms")
start_time = time.time()
result = binary_search(arr, x)
end_time = time.time()
print(f"Binary Search: Found at index {result}. Time taken: {(end_time - start_time) * 1000:.6f} ms")
##Linear Search: Time Complexity is O(n), where n is the number of elements in the list. Its performance degrades linearly with the size of the list.
##Binary Search: Time Complexity is O(log n), making it much more efficient for large datasets, but it requires the list to be sorted.





##bubble, insertion, selection sort algo
import time
def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
arr_sizes = [100, 1000]
algorithms = [bubble_sort, selection_sort, insertion_sort]
for size in arr_sizes:
    print(f"\nArray size: {size}")
    for algorithm in algorithms:
        test_arr = [i for i in range(size, 0, -1)]
        start_time = time.time()
        algorithm(test_arr)
        end_time = time.time()
        print(f"{algorithm.__name__}: {(end_time - start_time) * 1000:.6f} ms")
##Efficiency Comparison
##Bubble Sort: Its time complexity in the average and worst-case scenarios is O(n²), where n is the number of items being sorted. Due to its simplicity, it is not used for large datasets.
##Selection Sort: Like Bubble Sort, Selection Sort also has a time complexity of O(n²) for average and worst-case scenarios, making it inefficient for large lists.
##Insertion Sort: It also has an average and worst-case time complexity of O(n²). However, it works well for small lists or nearly sorted lists.





##recursion fibonacci,factorial,tower of hanoi
##Factorial
##Recursive Approach
def factorial_recursive(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial_recursive(n - 1)
##Iterative Approach
def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

##Fibonacci Sequence
##Recursive Approach
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
##Iterative Approach
def fibonacci_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

##Tower of Hanoi
def tower_of_hanoi(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    tower_of_hanoi(n-1, source, auxiliary, destination)
    print(f"Move disk {n} from {source} to {destination}")
    tower_of_hanoi(n-1, auxiliary, destination, source)
##Comparison of Recursive and Iterative Approaches
##Factorial and Fibonacci:
##Performance: Recursive solutions for factorial and Fibonacci are elegant but not always efficient, especially for large n values in Fibonacci, due to the high number of recursive calls and stack space usage. Iterative solutions are generally more efficient and should be preferred for large inputs.
##Readability: Recursive solutions are often more straightforward and easier to understand at a glance, provided one is familiar with recursion.
##Memory Usage: Iterative approaches are more memory-efficient than recursive ones because they do not involve the overhead of maintaining a call stack.
##Applicability: For factorial calculations, both approaches work well even for relatively large numbers (up to the point where integer overflow occurs if standard integer types are used). However, for the Fibonacci sequence, the recursive approach becomes impractical for relatively small n values due to exponential time complexity, unless memoization is used.





##N'th min max elmt
import heapq
def nth_min_max_sort(lst, n):
    sorted_lst = sorted(lst)
    nth_min = sorted_lst[n - 1]
    nth_max = sorted_lst[-n]
    return nth_min, nth_max
def nth_min_max_heap(lst, n):
    nth_min = heapq.nsmallest(n, lst)[-1]
    nth_max = heapq.nlargest(n, lst)[-1]
    return nth_min, nth_max
lst = [7, 2, 9, 1, 6, 3, 8, 5, 4]
n = 3
min_sort, max_sort = nth_min_max_sort(lst, n)
print(f"{n}th minimum element (Sorting Approach): {min_sort}")
print(f"{n}th maximum element (Sorting Approach): {max_sort}")
min_heap, max_heap = nth_min_max_heap(lst, n)
print(f"{n}th minimum element (Heap Approach): {min_heap}")
print(f"{n}th maximum element (Heap Approach): {max_heap}")
##Efficiency Comparison:
##Sorting Approach:
##Time Complexity: O(n log n) for sorting the list.
##Space Complexity: O(n) for storing the sorted list.
##Heap Approach:
##Time Complexity: O(n log k) where k is the size of the heap (in this case, k = n).
##Space Complexity: O(k) for the heap.





##general way & bruteforce
##General Pattern Matching Algorithm (Using Regular Expressions):
import re
def general_pattern_matching(text, pattern):
    matches = re.findall(pattern, text)
    return matches
text = "ababcabababc"
pattern = "ab"
print("General Pattern Matching:", general_pattern_matching(text, pattern))

##Brute Force Pattern Matching Algorithm:
def brute_force_pattern_matching(text, pattern):
    matches = []
    n = len(text)
    m = len(pattern)
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)
    return matches
text = "ababcabababc"
pattern = "ab"
print("Brute Force Pattern Matching:", brute_force_pattern_matching(text, pattern))
##Efficiency Comparison:
##General Pattern Matching (Using Regular Expressions):
##Time Complexity: O(n + m), where n is the length of the text and m is the length of the pattern.
##Space Complexity: O(k), where k is the number of matches found.
##Brute Force Pattern Matching:
##Time Complexity: O((n - m + 1) * m), where n is the length of the text and m is the length of the pattern.
##Space Complexity: O(k), where k is the number of matches found.





##fibonacci series, lcs, dynamic prog
##Fibonacci Series:
##Recursive Approach:
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
##Dynamic Programming Approach:
def fibonacci_dynamic(n):
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]
##Longest Common Subsequence (LCS):
##Recursive Approach:
def lcs_recursive(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs_recursive(X, Y, m - 1, n - 1)
    else:
        return max(lcs_recursive(X, Y, m, n - 1), lcs_recursive(X, Y, m - 1, n))
##Dynamic Programming Approach:
def lcs_dynamic(X, Y, m, n):
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]
# Fibonacci
n = 10
print("Fibonacci (Recursive):", fibonacci_recursive(n))
print("Fibonacci (Dynamic):", fibonacci_dynamic(n))
# Longest Common Subsequence
X = "AGGTAB"
Y = "GXTXAYB"
print("Longest Common Subsequence (Recursive):", lcs_recursive(X, Y, len(X), len(Y)))
print("Longest Common Subsequence (Dynamic):", lcs_dynamic(X, Y, len(X), len(Y)))
##Comparing Time Complexity:
##Fibonacci Series:
##Recursive Approach: Time Complexity - O(2^n)
##Dynamic Programming Approach: Time Complexity - O(n)
##Longest Common Subsequence:
##Recursive Approach: Time Complexity - O(2^(m+n))
##Dynamic Programming Approach: Time Complexity - O(m * n)





##merge sort, straseens mtrx multi DnC
##Merge Sort:
import time
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)
        merge_sort(right_half)
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
def print_array(arr):
    for i in arr:
        print(i, end=" ")
    print()
arr = [12, 11, 13, 5, 6, 7]
print("Original array:", arr)
start_time = time.time()
merge_sort(arr)
end_time = time.time()
print("Sorted array:", arr)
print("Time taken:", end_time - start_time, "seconds")
##Strassen's Matrix Multiplication:
import time
def strassen_matrix_multiply(A, B):
    if len(A) == 1:
        return [[A[0][0] * B[0][0]]]
    n = len(A) // 2
    A11, A12, A21, A22 = [], [], [], []
    B11, B12, B21, B22 = [], [], [], []
    for i in range(n):
        A11.append(A[i][:n])
        A12.append(A[i][n:])
        B11.append(B[i][:n])
        B12.append(B[i][n:])
    for i in range(n, len(A)):
        A21.append(A[i][:n])
        A22.append(A[i][n:])
        B21.append(B[i][:n])
        B22.append(B[i][n:])
    M1 = strassen_matrix_multiply(add_matrices(A11, A22), add_matrices(B11, B22))
    M2 = strassen_matrix_multiply(add_matrices(A21, A22), B11)
    M3 = strassen_matrix_multiply(A11, subtract_matrices(B12, B22))
    M4 = strassen_matrix_multiply(A22, subtract_matrices(B21, B11))
    M5 = strassen_matrix_multiply(add_matrices(A11, A12), B22)
    M6 = strassen_matrix_multiply(subtract_matrices(A21, A11), add_matrices(B11, B12))
    M7 = strassen_matrix_multiply(subtract_matrices(A12, A22), add_matrices(B21, B22))
    C11 = add_matrices(subtract_matrices(add_matrices(M1, M4), M5), M7)
    C12 = add_matrices(M3, M5)
    C21 = add_matrices(M2, M4)
    C22 = add_matrices(subtract_matrices(add_matrices(M1, M3), M2), add_matrices(M6, M7))
    result = []
    for i in range(len(C11)):
        result.append(C11[i] + C12[i])
    for i in range(len(C21)):
        result.append(C21[i] + C22[i])
    return result
def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def subtract_matrices(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
A = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]
B = [[17, 18, 19, 20],
     [21, 22, 23, 24],
     [25, 26, 27, 28],
     [29, 30, 31, 32]]
print("Matrix A:")
for row in A:
    print(row)
print("\nMatrix B:")
for row in B:
    print(row)
start_time = time.time()
result = strassen_matrix_multiply(A, B)
end_time = time.time()
print("\nResultant Matrix:")
for row in result:
    print(row)
print("\nTime taken:", end_time - start_time, "seconds")






##N-queen & bin str gen
##N-Queens Problem:
def is_safe(board, row, col, n):
    for i in range(col):
        if board[row][i] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    return True
def solve_n_queens(n):
    board = [[0] * n for _ in range(n)]
    if not solve_n_queens_util(board, 0, n):
        print("Solution does not exist")
        return False
    print_solution(board)
    return True
def solve_n_queens_util(board, col, n):
    if col >= n:
        return True
    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1
            if solve_n_queens_util(board, col + 1, n):
                return True
            board[i][col] = 0
    return False
def print_solution(board):
    for row in board:
        print(" ".join(map(str, row)))
solve_n_queens(4)
##Binary String Generation:
def generate_binary_strings(n):
    result = []
    generate_binary_strings_util(n, "", result)
    return result
def generate_binary_strings_util(n, current, result):
    if len(current) == n:
        result.append(current)
        return
    generate_binary_strings_util(n, current + "0", result)
    generate_binary_strings_util(n, current + "1", result)
print("Binary Strings:", generate_binary_strings(3))
# N-Queens Problem
print("N-Queens Problem:")
solve_n_queens(4)
# Binary String Generation
print("Binary Strings:")
print(generate_binary_strings(3))
##N-Queens Problem:
##Time Complexity: O(n!), where n is the size of the chessboard. This is because there are n choices for the queen's position in the first column, n-2 choices for the second column (after eliminating unsafe positions), n-4 choices for the third column, and so on. This results in a factorial time complexity.
##Binary String Generation:
##Time Complexity: O(2^n), where n is the length of the binary string. This is because at each position in the string, we have 2 choices (either 0 or 1), resulting in a binary tree-like structure with 2^n leaf nodes.

