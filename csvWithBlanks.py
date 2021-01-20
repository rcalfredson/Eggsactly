import csv

with open('myTest.csv', 'wt', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['','secondcell', 'e', '', 'fifthCell'])