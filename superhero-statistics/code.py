# --------------
#Code starts here
sc_df=data[['Strength','Combat']]
sc_covariance=sc_df.cov().iloc[0,1]
sc_strength=sc_df['Strength'].std()
sc_combat=sc_df['Combat'].std()
sc_pearson=sc_covariance/(sc_strength*sc_combat)

ic_df=data[['Intelligence','Combat']]
ic_covariance=ic_df.cov().iloc[0,1]
ic_intelligence=ic_df['Intelligence'].std()
ic_combat=ic_df['Combat'].std()
ic_pearson=ic_covariance/(ic_intelligence*ic_combat)



# --------------
#Code starts here
total_high=data['Total'].quantile(0.99)
super_best=data[data['Total'] > total_high]
super_best_names=list(super_best['Name'])
print(super_best_names)


# --------------
#Code starts here
import matplotlib.pyplot as plt
import pandas as pd
fig=plt.figure()
ax_1 = fig.add_subplot(1,1,1)
ax_1.boxplot(data['Intelligence'])
ax_1.set_title('Intelligence')
plt.show()

ax_2 = fig.add_subplot(2,1,1)
ax_2.boxplot(data['Speed'])
ax_2.set_title('Speed')
plt.show()

ax_3 = fig.add_subplot(1,1,1)
ax_3.boxplot(data['Power'])
ax_3.set_title('Power')
plt.show()






# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data=pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count=data['Gender'].value_counts()
plt.bar(gender_count,height=10)
#Code starts here 




# --------------
#Code starts here
alignment=data['Alignment'].value_counts()

label='Alignment'
plt.pie(alignment)



