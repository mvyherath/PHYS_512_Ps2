import numpy as np

def lorentz(x):
    lorentz.counter+=1
    return 1/(1+x**2)
lorentz.counter=0


# Explanation of this weird method I used to ensure the same values aren't used in f(x) is described at the bottom.


def integrate_adaptive(fun,x0,x1,tol):
    print('integrating between ',x0,x1)

    #hardwire to use simpsons
    x=np.linspace(x0,x1,5)
    y=fun(x)
    print(len(x), len(y))
    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    x_0.append(x0)
    x_1.append(x1)
    integrate_adaptive.count += 1
    print(err, tol)
 
    a = [elem - x0 for elem in x_0]

    if err<tol:
        return area2

    else:
        # This is where the restarting loop begins
        restart=True
        while restart:
            checker = None
            for elem in range(1, len(x_0), 1):
                diff = x_0[elem] - x0
                if diff == 0:
                    checker = True
                else:
                    checker = False

            if checker == True:
                restart = True
                x0 = (x0 + x1) / 2
            else:
                restart = False
                x0 = x0

        xmid=(x0+x1)/2                              # LINE X
        left=integrate_adaptive(fun,x0,xmid,tol/2)
        right=integrate_adaptive(fun,xmid,x1,tol/2)
        return left+right

        # This is where the restarting loop ends.


integrate_adaptive.count = 0
x_0 = [] #array to hold x0
x_1 = [] #array to hold x1


x0=-100
x1=100

ans=integrate_adaptive(lorentz,x0,x1,1e-7)
print(ans-(np.arctan(x1)-np.arctan(x0)))
print(lorentz.counter)


# I'm not sure if I understood the problem correctly. 
#So I coded as I understood what the problem is asking. I adjusted the adaptive integrated #in such a way that it would not run the function over the SAME values of x0 and x1 in any #subsequent step. So after the else condition (see above, I've marked where the changes #begin), I started a condition called "restart". Inside the "restart" condition, there is #a loop that runs through a list of values I've stored for x0. This list contains every x0 #value used by the adaptive integrator function. We loop through this list, comparing the #current value of x0 to all other stored values of x0 in the list. The current x0 is #reduced from each of the elements in the list of x0 values. If the current x0 is equal to #any of the values in the list, then x_0 [elem] - x0 = 0. I put a checker inside the #restart condition. If x_0[elem] - x_0 = 0, then the checker is set to True. If it is not #equal, it is set to False. Subsequently, if the Checker is True, then restart = True. If #restart = True, then the current value of x0 is altered using x0 = (x0 + x1) /2. Then #because the restart condition is True, we ONCE AGAIN loop through the list of x0 values. #This time, if the new x0 value is not equal to any of the elements of the x_0 list, then #checker is set to False, and therefore the loop does not restart again. Instead it goes #straight to "LINE X" (marked with a hashtag above). This reduced the number of steps #required to reach the answer. I'm aware that this method probably doesn't give a correct #answer, but I wanted to somehow write the code to by-pass as many of the x0 values that #were used before. I'm a bit confused about where to go from here because the code for the #adaptive integrator itself is going to take a bit of time to understand. Using a #"restart" condition seemed like the best way for me to at least reduce the number of function calls.  

