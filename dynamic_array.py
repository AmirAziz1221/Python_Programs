dynamic_arr = []
#add elements 
dynamic_arr.append(15)
dynamic_arr.append(22)
dynamic_arr.append(33)
print(dynamic_arr[0])
print(dynamic_arr[1])
print(dynamic_arr[2])
#remove
dynamic_arr.remove(22)
print("after remove: ",dynamic_arr[1])
#lenth of the array
length = len(dynamic_arr)
print(length)
#add more Elements
dynamic_arr.append(12)
dynamic_arr.append(50)
dynamic_arr.append(90)
print("Complete array ")
for item in dynamic_arr:
    print(item)
length = len(dynamic_arr)
print(length)