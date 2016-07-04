from wordcloud import WordCloud
import matplotlib.pyplot as plt

line='0.475*director + 0.302*band + 0.169*school + 0.014*student + 0.011*ingredient + 0.000*aicf + 0.000*development + 0.000*foundation + 0.000*parent + 0.000*life'
scores = [x.split("*")[0] for x in line.split(" + ")]
words = [x.split("*")[1] for x in line.split(" + ")]
freqs = []
w=[]
curr_topic = 0

for word, score in zip(words, scores):
    freqs.append((word, float(score)))
    w.append(word)
wc=WordCloud()
cloud=wc.fit_words(freqs)
plt.imshow(cloud)
plt.axis("off")
plt.show()