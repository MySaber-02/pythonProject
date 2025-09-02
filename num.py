import numpy as np
import pandas as pd
# import numpy as np
# matrix=np.array([[10,20,30],[40,50,60],[70,80,90]])
# print(matrix[:,1])
# print(matrix[:,0:2])
# print(matrix[1:3,:])
# print(matrix[1:3,0:2])

# import numpy as np
# matrix=np.array([[10,20,30],[40,50,60],[70,80,90]])
# second_column_50=(matrix[:,1]==50)
# print(second_column_50)
# print(matrix[second_column_50,:])

# import numpy
# vector =numpy.array([10,20,30,20])
# equal_to_ten_or_five=(vector==20)|(vector==20)
# print(vector[equal_to_ten_or_five])
# vector[equal_to_ten_or_five]=200

# print(vector)

# import numpy as np
 
# matrix=np.array([[10,20,30],[40,50,60],[70,80,90]])
# print(matrix)
# print(matrix.sum(axis=0))
# print(matrix.sum(axis=1))
 
# print(np.array([5,10,20]))
 

 
# print(np.array([10,10,15]))

# df1=pd.DataFrame(np.arange(9).reshape(3,3),columns=list('bcd'),index=['b','s','g'])
# print(df1)


# def plus(df,n):
 
#     df['c'] = (df['a']+df['b'])
    
#     df['d'] = (df['a']+df['b']) * n
    
#     return df
 
# list1 = [[1,3],[7,8],[4,5]]
 
# df1 = pd.DataFrame(list1,columns=['a','b'])
 
# df1 = df1.apply(plus,axis=1,args=(2,))
 
# print(df1)

df1 = pd.DataFrame(np.arange(9).reshape(3,3),columns=list('bcd'),index=['b','s','g'])
print(df1)
df2 = pd.DataFrame(np.arange(12).reshape(4,3),columns=list('cde'),index=['b','s','g','t'])
print(df2.drop('c'))
df1+df2
 
df3 = df1.add(df2,fill_value='0')
df4 = df1.sub(df2,fill_value='0')
df5 = df1.mul(df2,fill_value='1')
df6 = df1.div(df2,fill_value='1')
df7 = df1.subtract(df2,fill_value='0')
df8 = df1.multiply(df2,fill_value='1')