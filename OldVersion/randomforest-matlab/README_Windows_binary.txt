This readme supercedes the ones in the sub-folders if you plan to use just the binaries on a Windows machine.
The binary for windows are already included, and should work out of the box.

Matlab usually installs these redistributable on Installation, but ...

But in case they dont work. Open the control panel and check for the following package:
      Microsoft visual C++ 2005 redistributable (x64 if your machine is 64-bit or x86 for 32-bit OS)

If either is not there, you only need the 32-bit if you plan to use it on a 32-bit machine and same goes for 64-bit.

32-bit redistributable:
http://www.microsoft.com/downloads/details.aspx?familyid=32bc1bee-a3f9-4c13-9c99-220b62a191ee&displaylang=en

64-bit redistributable:
http://www.microsoft.com/downloads/details.aspx?familyid=90548130-4468-4BBC-9673-D6ACABD5D13B&displaylang=en

Why do you need this?
Well, the package was compiled with Matlab and Visual C++ express edition. I reckon the latter might not be installed
on your machine. This is where the redistributable help. They contain the necessary libraries, dll to make the mex run.
Matlab should install these redistributables when its installed but you never know...