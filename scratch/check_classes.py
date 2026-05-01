import pickle
import os

if os.path.exists("class_names.txt"):
    with open("class_names.txt", "r") as f:
        classes = f.read().splitlines()
        print("CLASSES FOUND:")
        for c in classes:
            print(f"- {c}")
else:
    print("class_names.txt NOT FOUND")
