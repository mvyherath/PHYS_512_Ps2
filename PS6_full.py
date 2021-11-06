# Standard python numerical analysis imports:
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# LIGO-specific readligo.py 
import readligo as rl
from scipy.signal import medfilt

# NOTE: OUTPUTS in PS6_Outputs.txt
# NOTE: COMMENTS FOR PARTS d), e) and f) in PS6_Comments.txt


#------------------------The median filter--------------------------------------

# The medfilt function performs a median filter on an N dimensional array
# It takes an odd number of points and takes the median for that region....
#..... using those points. We use 9 points because this was found to be....
#..... optimum through trial and error
# This function will be used to smooth the noise model


def MedianFilter(x, kernel_size=9):
    
    "A wrapper for the default Python median filter"

    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(x, kernel_size=kernel_size)


#------------------------The Window Function------------------------------------

# The Scipy.Tukey Window Function is used in this project


def window(npoints):
    return signal.tukey(npoints)




# We put all the event names inside an array
eventname = ['GW150914', 'GW151226', 'LVT151012', 'GW170104']

# Read the event properties from a local json file
fnjson = "BBH_events_v3.json"
events = json.load(open(fnjson,"r"))



# --------------------------PS6 PARTS a) and b)---------------------------------

# We loop through the array of event names

for i in range(len(eventname)):

    # Extract the parameters for the desired event:
    event = events[eventname[i]]
    fn_H1 = event['fn_H1']              # File name for H1 data
    fn_L1 = event['fn_L1']              # File name for L1 data
    fn_template = event['fn_template']  # File name for template waveform
    fs = event['fs']                    # Set sampling rate
    tevent = event['tevent']            # Set approximate event GPS time
    fband = event['fband']              # frequency band for bandpassing signal
    print("Reading in parameters for event " + event["name"])

    # read in data from H1 and L1, if available:
    # Getting strain
    strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
    strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')

    # both H1 and L1 will have the same time vector, so:
    time = time_H1
    # the time sample interval (uniformly sampled!)
    dt = time[1] - time[0]
    # Array to hold L1 and H1 strains
    strains = [strain_L1, strain_H1]

    # Getting template
    f_template = h5py.File(fn_template, "r")
    template_p, template_c = f_template["template"][...]

    # A 2nd loop is created to loop through L1 and H1
    names = ['L1', 'H1']
    for y in range(len(strains)):
        
    
        # Creating the Window Function 
        # The Fourier Transform of the strain x window func
        # Then we create the noise model Nft
        # Then we plot that noise model
        win = window(len(time)) 
        sft = np.fft.rfft(win * strains[y])
        Nft = np.abs(sft)**2

        # We use the MedianFilter func to smooth out the Noise template....
        #.... using a median filter. We use 9 points

        Nft = MedianFilter(Nft)

        # The next step is to whiten the noise template and the data
        # To whiten, we divide the noise and data templates by the sqrt of....
        #..... the smoothed noise model
        # This is done for both L1 and H1 for all events

        # smoothed noise template
        # data template whitening followed by Fourier transforming
        sft_white = sft / np.sqrt(Nft)  
        tft_white = np.fft.rfft(template_p * win) / np.sqrt(Nft)

        # getting the inverse Fourier transform of the data template
        t_white = np.fft.irfft(tft_white)

        # The MatchFilter (MF) is created by taking the conjugate of the FFT...
        #....of the whitened data template, multiplied by the FFT of the......
        #.... noise model, followed by the inverse FFT of the product
        # We also keep a non inverse FFT'd product of sft * conj(tft)

        MF = np.fft.irfft(sft_white * np.conj(tft_white))
        non_inv = sft_white * np.conj(tft_white)  # non inverse FFT'd product

        
# ----------------------------PS6 PART c)---------------------------------------


        # Noise calculation for all events
        # We use (sum(data[elems] - average)^2 / data points) ^ 0.5
        # NOTE - Livingston + Harford calculations are done right at the bottom
        
        average = np.mean(MF)
        delta_vals_sq = (MF - average)**2
        Noise = np.sqrt(np.sum(delta_vals_sq) / len(delta_vals_sq))
        print("Noise of"," ",names[y]," ","for event"," ",eventname[i]," ","=", Noise)

        

# ----------------------------PS6 PART d)---------------------------------------

        # Numerical SNR
        # To find the numerical SNR we take the MF with the noise model.......
        #....and we focus on the event spike. We take the maximum value at....
        #....the spike and the minimum value, and then add them together (H).
        # We take the standard deviation (std) of the data points that are....
        #....in the region of the first 20,000 data points. The reason is that..
        #....this is a distance that was determined to be far enough from......
        #....the event spike so that it would be unaffected, but also contains..
        #....enough data points to get a decent value of the std of the flat....
        #....parts of the data.
        # Then we divide H by the std of MF[:20000] to get the numerical SNR

        max_MF = np.abs(np.max(MF))
        min_MF = np.abs(np.min(MF))
        H = np.abs(max_MF + min_MF)   # height of the event spike

        region = MF[:20000]
        reg_std = np.std(region)      # std of the flat region away from the spike
        num_SNR = H / reg_std         # numerical SNR
        print("Numerical SNR of"," ",names[y]," ","for event"," ",eventname[i]," ","=", num_SNR)


        # Analytic SNR
        # To find the analytical SNR we take the MF_new with the data template......
        #...where the new MF is made by taking the product of the...............
        #...data template with the conjugate of itself,.........................
        #....and we focus on the event spike. We take the maximum value at....
        #....the spike (H_d).
        # We take the standard deviation (std) of the data points that are....
        #....in the region of the first 4000 data points. The reason is that..
        #....this is a distance that was determined to be far enough from......
        #....the event spike so that it would be unaffected, but also contains..
        #....enough data points to get a decent value of the std of the flat....
        #....parts of the data.
        # Then we divide H by the std of MF[:4000] to get the analytical SNR

        # MatchFilter with just product of the data template with conj(itself)

        MF_new = np.fft.irfft(tft_white * np.conj(tft_white))
        ini_pts = MF_new[:4000]    # flat region of new MF
        std_ini = np.std(ini_pts)
        maxi = np.max(MF_new)    # event spike of the new MF
        analytic_SNR = maxi / std_ini

        print("Analytical SNR of"," ",names[y]," ","for event"," ",eventname[i]," ","=", analytic_SNR)




# ----------------------------PS6 PART e)---------------------------------------

        # Finding frequency where half the weghts come from below and above
        # We first use the sampling rate of 4096, the frequency which is the.....
        #....inverse of it.
        # Each event is 32 seconds long. So we take a number of points that.....
        #....is splititng 32 seconds by the pts in the Match Filter.
        # We derive the frequncies using fft.fftfreq on the num of pts found....
        #....by splitting 32 seconds with num_pts(Match Filter).
        # Then we find df = freq[2nd] - freq[1st]
        # We find the area underneath the Match Filter with................
        # Area = sum (MF_vals * df), where MF_vals is the Fourier Transform of..
        #....the initial Match Filter.
        # Basically the method here is that we find the area under the MF, and..
        #....then find the frequncy at which we get HALF the area under the MF.
        # That would be the frequncy at which the weghts are halved.

        sr = 4096  # sampling rate
        points = np.linspace(0, 32, len(MF))
        f_vals = np.fft.fftfreq(len(points), 1/sr)
        df = f_vals[1] - f_vals[0]

        area = np.sum(np.abs(np.fft.rfft(MF)) * df)
        half_area = area / 2

        # Knowing the total area, we test different numbers of data points from..
        #....MF until the variable "area2" gives a value = half the area under..
        #....the MF. This was done via trial and error because of complications..
        #....that came up by trying to loop.

        area2 = np.sum(np.abs(np.fft.rfft(MF[:85580])) * df)
        # We get the frequncy corresponding to the element of 85580
        # The value of 85580 is fairly constant across events
        frequency = np.abs(f_vals[85580])
        print("Frequncy for"," ",names[y]," ","for event"," ",eventname[i]," ","=", frequency)



        # Plotting the Noise model and two match filters
        
        fig, axs = plt.subplots(3)
        fig.suptitle('Noise model and two match filters of' + " " + names[y] + " " + 'for event' + " " + eventname[i])
        axs[0].loglog(Nft, color='blue')
        plt.ylabel('Noise')

        axs[1].plot(np.fft.fftshift(MF), color='green')
        plt.ylabel('Data')

        axs[2].plot(np.abs(np.fft.fftshift(MF_new)), color='red')
        plt.xlabel('Time')
        plt.ylabel('Data')

        plt.show()

        
        
        
    # Calculations for Livingston + Hanford events

    strain_aggr = (strain_H1 + strain_L1) / 2  # aggregate of L1 and H1 detectors
    win_aggr = window(len(time))
    sft_aggr = np.fft.rfft(win_aggr * strain_aggr)
    Nft_aggr = np.abs(sft_aggr)**2
    Nft_aggr = MedianFilter(Nft_aggr)

    sft_white_aggr = sft_aggr / np.sqrt(Nft_aggr)
    tft_white_aggr = np.fft.rfft(template_p * win_aggr) / np.sqrt(Nft_aggr)
    t_white_aggr = np.fft.irfft(tft_white_aggr)
    # Match Filter for combined detector strains
    MF_aggr = np.fft.irfft(sft_white_aggr * np.conj(tft_white_aggr))

    # Noise calculation
    average_aggr = np.mean(MF_aggr)
    delta_vals_sq_aggr = (MF_aggr - average_aggr)**2
    Noise_aggr = np.sqrt(np.sum(delta_vals_sq_aggr) / len(delta_vals_sq_aggr))
    print("Noise of Livingston + Harford for", eventname[i], "=", Noise_aggr)

    # Numerical SNR
    max_MF_aggr = np.abs(np.max(MF_aggr))
    min_MF_aggr = np.abs(np.min(MF_aggr))
    H_aggr = np.abs(max_MF_aggr + min_MF_aggr)

    region_aggr = MF_aggr[:20000]
    reg_std_aggr = np.std(region_aggr)

    num_SNR_aggr = H_aggr / reg_std_aggr
    print("Numerical SNR of Livingston + Harford for", eventname[i], "=", num_SNR_aggr)

    


print("Ending")
