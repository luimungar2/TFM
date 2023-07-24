import re
import os, sys
import subprocess
app = SmaliscaApp()
app.setup()
#Set log level
app.log.set_level('info')
path = "C:/Users/usuario/.spyder-py3/ciber/TFM/stage2/stage/"
dirs = os.listdir( path )
root="C:/Users/usuario/.spyder-py3/ciber/TFM/stage2/stage/"
i=0
location = 'C:/Users/usuario/.spyder-py3/ciber/TFM/stage2/stage/'
#Specify file name suffix
suffix = 'smali'
#Create a new parser
parser = SmaliParser(location, suffix)
parser.run()
results = parser.get_results()
#results1=re.findall(r'to_ method \ ':\'(.*?)\'\}',str(results))
results1 = re.findall(r"'to_method':\s*'([^,]*)'", str(results))
#results2=re.sub('\ '',"",str(results1))
results2 = re.sub("\ '", "", str(results1))
c=['startService','getDeviceId','createFromPdu','getClassLoader',
'getClass','getMethod','getDisplayOriginatingAddress',
'getInputStream','getOutputStream','killProcess',
'getLine1Number','getSimSerialNumber','getSubscriberId',
'getLastKnownLocation','isProviderEnabled']
b=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#print(results)
for C in c:
	#if re.search(r ' ' +C, str(results1)):
    if C in str(results1):
        b[i]=1
        print("Permiso encontrado: ",i,C)
    else:
        print("Permiso no encontrado: ",i,C)
    i=i+1
print("\nResumen de la matriz de coincidencia",b)
print("\nPorcentaje de coincidencia de permisos según el criterio: ",str((sum(b)/len(b))*100)+"%")
