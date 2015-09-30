from __future__ import print_function

# import your hw0.py solution
import hw0

points = 0
total = 2

print('Testing Problem 1')

name, email, completed = hw0.prob1()

if name != "YOUR NAME" and email != "yourname@stanford.edu" and completed is True:
    print('Pass')
    points += 1
else:
    print('Fail')


print('Testing Problem 2')

if hw0.prob2(0, 0) == 0 and hw0.prob2(5, 17) == 22 and hw0.prob2(.1, -10) == -9.9:
    print('Pass')
    points += 1
else:
    print('Fail')

print('Passed {} out of {} questions'.format(points, total))
 