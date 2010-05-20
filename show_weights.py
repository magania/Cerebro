#! python

f=open('W_1.txt', 'r')

nline=0
for line in f.readlines():
  sW=line.split()
  W=[]

  for w in sW:
    W.append(float(w))

  w_max = max(W)
  w_min = min(W)
  w_size = w_max-w_min

  iW = []
  for w in W:
    iW.append(int(255*(w-w_min)/w_size))

  out = open('img/' + str(nline) + '.ppm', 'w')
  out.write('P3 28 28 255\n')
  count=0
  for w in iW:
    out.write('0 0 ' + str(w) + '  ')
    if count % 28 == 27:
      out.write('\n')
    count = count+1

  out.close()

  nline = nline+1
