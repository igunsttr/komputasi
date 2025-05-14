import matplotlib.pyplot as plt

# Data
labels = 'Python', 'C++', 'Java', 'Julia', 'Scala'
sizes = [215, 130, 245, 210, 310]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']
explode = (0, 0, 0.1, 0, 0)  # explode 1 slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  

plt.show()