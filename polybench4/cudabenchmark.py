import os
import re
import time
import threading
#import matplotlib.pyplot as plt
import subprocess
import filecmp
import multiprocessing

class ParserException(Exception):
   def __init__(self, message):
      Exception.__init__(self, message)

def extract_kernel_timings(profout):
   re_str = r'\s*(\d*.\d+)[mu]*s\s*'
   unit_re_str = r'([mu]*s)'
   time_regex = re.compile(re_str)
   unit_regex = re.compile(unit_re_str)
   total_time = 0.0 
   unit = ""
   
   nmatched = 0

   for line in profout.split("\n"):
      if 'kernel' not in line:
         continue
      line = line.strip()
      matches = time_regex.findall(line)
      #matches[0] has the total time 
      if matches:
         nmatched += 1
         try:
            val = float(matches[0])
         except:
            raise ParserException("Not able to extract execution time from "+line)
         unit = unit_regex.findall(line)[0]
         if unit == 'us':
            val = val / 1000
         elif unit == 's':
            val = val * 1000
         total_time += val
   if nmatched == 0 :
      raise  ParserException( "Regular exception failed to extract time "+re_str)
   
   print("Execution time %.4f " %total_time+unit)
   return total_time

def allbenchmarks(fname):
   ls = []
   with open(fname, 'r') as blist:
      for b in blist:
         ls.append(b)
   return ls 

def median(ls):
   num = len(ls)
   ls.sort()
   middle = num/2
   return ls[middle]

def clean(name):
   dir = os.path.dirname(name)
   proc = subprocess.Popen('make clean', cwd=dir, shell=True, stderr=subprocess.PIPE)
   stdout, stderr = proc.communicate()

def compile_bench(name):
   dir = os.path.dirname(name)
   proc = subprocess.Popen('make clean', cwd=dir, shell=True, stderr=subprocess.PIPE)
   stdout, stderr = proc.communicate()
   proc = subprocess.Popen('make compile', cwd=dir, shell=True, stderr=subprocess.PIPE)
   stdout, stderr = proc.communicate()
   return name

def isSchedDiff(name):
   dir = os.path.dirname(name)
   ppcg_sched = dir+'/sched_ppcg'
   spat_sched = dir+'/sched_spat'
   return not filecmp.cmp(ppcg_sched, spat_sched)

def runbenchmark(name):
   ppcg_time = []
   spat_time = []
   dir = os.path.dirname(name)
   for i in range(1):
      proc = subprocess.Popen('make cuda', cwd=dir, shell=True, stderr=subprocess.PIPE)
      stdout, ppcg_prof = proc.communicate()
      proc = subprocess.Popen('make cuda_endsgrp', cwd=dir, shell=True, stderr=subprocess.PIPE)
      stdout, spat_prof = proc.communicate()
      ppcg_time.append(extract_kernel_timings(ppcg_prof))
      spat_time.append(extract_kernel_timings(spat_prof))
      speedup = median(ppcg_time) / median(spat_time)
      with open('speedups.d', 'a') as fp: 
         fp.write(os.path.basename(name))
         fp.write(str(speedup))

   return speedup 


#extract_kernel_timings("prof")
benchmarks = allbenchmarks('./benchmark_list')
pool = multiprocessing.Pool()
pool.map(compile_bench, benchmarks)
#map(compile_bench, benchmarks)
diff_benchmarks = list(filter(isSchedDiff, benchmarks))
print(diff_benchmarks)
with open('sched_diff', 'w') as fp:
   fp.write("\t".join(diff_benchmarks))
speedups = list(map(runbenchmark, benchmarks))
names = list(map((lambda b: os.path.basename(b)), diff_benchmarks))
with open('speedups.dump', 'w') as fp:
   fp.write("\t".join(names))
   fp.write("\n".join(str(e) for e in speedups))

#xvals = range(len(speedups))
#plt.bar(xvals, speedups)
#plt.ylabel('speedup')
#plt.xlabel('benchmarks')
#plt.tight_layout()
#plt.xticks(xvals, names)
#plt.savefig('speedups.png')
#plt.show()
#forallbenchmarks('./benchmark_list')

