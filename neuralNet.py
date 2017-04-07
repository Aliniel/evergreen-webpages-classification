import numpy as np
import neurolab as nl
import pylab as pl
import pandas

train_df = pandas.DataFrame(pandas.read_csv('D:\processed_train.tsv', sep='\t'))

#for key in train_df:
#    print(key)

size = 100

x = np.zeros((size,5));
y = np.zeros(size);

#for key in train_df:
 #   print(key)

for i in range(0,size):
  x[i][0] = float(train_df['common_link_ratio_2'][i])
  x[i][1] = float(train_df['html_ratio'][i])
  x[i][2] = float(train_df['image_ratio'][i])
  x[i][3] = float(train_df['has_long_links'][i])
  x[i][4] = float(train_df['spelling_errors_ratio'][i])

  y[i] = float(train_df['is_evergreen'][i])

y = y.reshape(size,1)

net = nl.net.newff([[0,1]]*5, [5,1])

error = net.train(x, y, epochs=50, show=10)

size2 = 50

z = np.zeros((size2,5));
q = np.zeros(size2);
for i in range(0,size2):
  z[i][0] = float(train_df['common_link_ratio_2'][i+size])
  z[i][1] = float(train_df['html_ratio'][i+size])
  z[i][2] = float(train_df['image_ratio'][i+size])
  z[i][3] = float(train_df['has_long_links'][i+size])
  z[i][4] = float(train_df['spelling_errors_ratio'][i+size])

  q[i] = float(train_df['is_evergreen'][i+size])

out = net.sim(z)
#out.reshape(size2)
#q.reshape(size2)

pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')

pl.subplot(212)
pl.plot(out,'r',q,'g')
pl.show()