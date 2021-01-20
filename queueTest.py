import queue
myQueue = queue.LifoQueue(maxsize=0)
myQueue.put('item1')
myQueue.put('item2')
myQueue.put('item3')
myQueue.put('item1')
while not myQueue.empty():
    print('got the item', myQueue.get())