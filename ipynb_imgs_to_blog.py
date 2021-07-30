def h(s):
    r = s[s.find("(")+1:s.find(")")]
    u, c = r.split(" ", 1)
    u = u.replace("my_icons", "images")
    st = f'''
<center>
<img src="{{{{site.baseurl}}}}/{u}" style="zoom: 70%;"/><br>
<figcaption><i>{c[1:len(c)-1]}</i></figcaption>
</center>
    '''
    return st

def g(s):
    u = s[s.find("(")+1:s.find(")")]
    st = f'''
<center>
<img src="{{{{site.baseurl}}}}/images/2020/07/mnist-qnn/{u}" style="zoom: 70%;"/>
</center>
    '''
    return st

filename = "2020-07-14-mnist-qnn.md"
filename2 = filename + "2"

with open(filename, 'r') as f:
    content = f.readlines()

f.close()

for idx, line in enumerate(content):
    if 'my_icons' in line:
        content[idx] = h(line)
        print(content[idx])
    if '![png]' in line:
        content[idx] = g(line)
        print(content[idx])

with open(filename2, 'w') as f:
    f.writelines(content)
