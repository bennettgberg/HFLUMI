import sys, os
from math import exp,log
import argparse
import subprocess
import ROOT
import array
import tables
import glob
import numpy as np
import multiprocessing as mp
import time 

tot_start = time.time()
parser=argparse.ArgumentParser()
parser.add_argument("-f", "--onefile", default="", help="The path to a cert tree or root file with \"Before\" histograms.")
parser.add_argument("-o", "--out", default="SBR_Opt", help="Outputname ")
parser.add_argument("-m", "--method", default="et", help="method ")
parser.add_argument("-d", "--dir",  default="", help="The path to a directory of cert trees or roots file with \"Before\" histograms.")
parser.add_argument( '--makeplot', help='makeplots', default=False,action="store_true")
parser.add_argument("-w", "--write", help="write new HFSBR to file and exit", default=False, action="store_true") 
parser.add_argument("-ow", "--outwrite", help="what file to write out the new HFSBR to", default="HFSBR_ET_22v12.txt") 
parser.add_argument("-pc", "--processingcores", help="how many cores to use for multiprocessing", default=1, type=int) 
parser.add_argument("-pf", "--paramfile", help="file to read in starting parameters from", default="etparams.txt") 
parser.add_argument("-xp", "--xparam", help="which parameter index to modify as the 'x' variable", default=-1, type=int) 
parser.add_argument("-yp", "--yparam", help="which parameter index to modify as the 'y' variable", default=-1, type=int) 

args=parser.parse_args()

dooccupancy=False
if args.method == 'oc':
    dooccupancy=True

dataQueue = mp.Queue()
write_file = False
if args.write and args.processingcores == 1: write_file = True
if args.dir=="":
    hfRawHD5FileName = args.onefile
    print(hfRawHD5FileName)
    hdf = tables.open_file(hfRawHD5FileName)
    hists=[]
    if dooccupancy:
        hists.append([hdf.root.hfCMS_VALID,hdf.root.hfCMS1])
    else:
        hists.append([hdf.root.hfCMS_VALID,hdf.root.hfCMS_ET])
else:
    if args.onefile!="":
        filelist=glob.glob(args.dir+'/'+args.onefile)
    else:
        filelist=glob.glob(args.dir+'/*hd5')
    hists=[]
    for ifile in filelist:
        print(ifile)
        hdf = tables.open_file(ifile)
        if dooccupancy:
            hists.append([hdf.root.hfCMS_VALID,hdf.root.hfCMS1])
        else:
            hists.append([hdf.root.hfCMS_VALID,hdf.root.hfCMS_ET])

thisLNHists={}
thisLNHists['hfCMS_VALID']={}
thisLNHists['hfCMS_ET']={}
thisLNHists['hfCMS1']={}
thisLNHists['mu']={}

tot_res_={} 

HFSBR=[]
##3564
MAX_NBX=3564
MAX_NBX_NOAG=3480

# Test optimization
thisLN=0
lastLN=-1
# Read SBR
if dooccupancy:
    #text_file = open("HFSBR_OC.txt", "r")
    text_file = open("HFSBR_OC_22v3.txt", "r")
    HFSBR = text_file.read().split(',')
    text_file.close()
    for i in range(len(HFSBR)):
        HFSBR[i]=float(HFSBR[i])
else:
    text_file = open("HFSBR.txt", "r")
    HFSBR = text_file.read().split(',')
    text_file.close()
    for i in range(len(HFSBR)):
        HFSBR[i]=float(HFSBR[i])
HFSBR_new=np.copy(HFSBR)

def MakeDynamicBunchMask(rawdata):
    rawdata=np.array(rawdata)
    minv=1
    maxv=0
    activeBXMask=np.zeros(MAX_NBX)
    NActiveBX=0
    maxBX=0
    if dooccupancy:
        lumiAlgo="occupancy"
    else:
        lumiAlgo="etsum"
    maxBX=np.argmax(rawdata[0:MAX_NBX_NOAG])
    maxv=rawdata[maxBX]
    abMin=1e-3 if lumiAlgo.find("etsum") != -1 else 8e-6
    minv= max( 0 if np.amin(rawdata[0:MAX_NBX_NOAG]) == 1 else np.amin(rawdata[0:MAX_NBX_NOAG]), abMin)
    nearNoise=1e-5
    aboveNoise=1e-4
    fracThreshold=0.2
    if lumiAlgo.find("etsum") !=-1 :
        nearNoise = 0.02
        aboveNoise = 0.05
    if  maxv-minv<nearNoise :
        fracThreshold=1.
    elif (maxv-minv<aboveNoise) :
        fracThreshold= (aboveNoise-maxv+minv)/(aboveNoise-nearNoise) + (maxv-minv-nearNoise)*fracThreshold / (aboveNoise-nearNoise)
    dynamicThreshold=max((maxv-minv)*fracThreshold,abMin)
    if maxv-minv > 0.5*minv :
        activeBXMask = np.where(rawdata> minv + dynamicThreshold,1,0)
        activeBXMask[3481:3500] = 0
        NActiveBX = np.count_nonzero(activeBXMask==1)
    return [rawdata, minv, maxv, fracThreshold, dynamicThreshold, activeBXMask, NActiveBX]

def ComputeAfterglow(rawdata, activeBXMask, HFSBR):
    rawdata_Afterglow=np.copy(rawdata)
    HFSBR=np.array(HFSBR)
    HFSBR[0]=0
    mask_residual_type1=np.ones(len(activeBXMask))
    mask_residual_type2=np.copy(activeBXMask)
    mask_residual_type2_cut5=np.copy(activeBXMask)
    mask_residual_type2_cut10=np.copy(activeBXMask)
    mask_residual_type2_cut50=np.copy(activeBXMask)
    for ibx in np.argwhere(activeBXMask==1):
        if activeBXMask[ibx+1]!=1:
             mask_residual_type1[ibx+1]=0
             mask_residual_type2[ibx+1]=1
             for ibxnext in range(5):
                  if activeBXMask[ibx+ibxnext+1]==1: break
                  mask_residual_type2_cut5[ibx+ibxnext+1]=1
             for ibxnext in range(10):
                  if activeBXMask[ibx+ibxnext+1]==1: break
                  mask_residual_type2_cut10[ibx+ibxnext+1]=1
             for ibxnext in range(50):
                  if activeBXMask[ibx+ibxnext+1]==1: break
                  mask_residual_type2_cut50[ibx+ibxnext+1]=1
        rawdata_Afterglow -= rawdata_Afterglow[ibx]*np.roll(HFSBR, ibx)
    # 3481 to 3500 is board gap
    mask_residual_type1[3481:3500]=1
    mask_residual_type2[3481:3500]=1
    mask_residual_type2_cut5[3481:3500]=1
    mask_residual_type2_cut10[3481:3500]=1
    mask_residual_type2_cut50[3481:3500]=1
    totalMuUn=np.sum(rawdata[np.where(activeBXMask==1)])
    totalMuCorr=np.sum(rawdata_Afterglow[np.where(activeBXMask==1)])
    hfAfterGlowTotalScale = totalMuCorr/totalMuUn if totalMuUn >0 else 0
    if hfAfterGlowTotalScale<0.5:
        print("The total correction factor is:  ",hfAfterGlowTotalScale)
    # average residual square
    residual_type1_AfterGlow = np.sum(np.square(rawdata_Afterglow[np.where(mask_residual_type1==0)])) / len(np.where(mask_residual_type1==0))
    residual_type2_AfterGlow = np.sum(np.square(rawdata_Afterglow[np.where(mask_residual_type2==0)])) / len(np.where(mask_residual_type2==0))
    residual_type2_cut5_AfterGlow = np.sum(np.square(rawdata_Afterglow[np.where(mask_residual_type2_cut5==0)])) / len(np.where(mask_residual_type2_cut5==0))
    residual_type2_cut10_AfterGlow = np.sum(np.square(rawdata_Afterglow[np.where(mask_residual_type2_cut10==0)])) / len(np.where(mask_residual_type2_cut10==0))
    residual_type2_cut50_AfterGlow = np.sum(np.square(rawdata_Afterglow[np.where(mask_residual_type2_cut50==0)])) / len(np.where(mask_residual_type2_cut50==0))
    return [ rawdata_Afterglow,hfAfterGlowTotalScale, totalMuUn], [mask_residual_type1, mask_residual_type2, mask_residual_type2_cut5, mask_residual_type2_cut10, mask_residual_type2_cut50], [residual_type1_AfterGlow, residual_type2_AfterGlow, residual_type2_cut5_AfterGlow, residual_type2_cut10_AfterGlow, residual_type2_cut50_AfterGlow]

def SubtractPedestal(rawdata, mask_residual,activeBXMask, totalMuUn):
    rawdata_AfterPed=np.copy(rawdata)
    pedestal = np.array([np.average(rawdata_AfterPed[np.arange(3500+0, 3500+52, 4, dtype=int)]),np.average(rawdata_AfterPed[np.arange(3500+1, 3500+52, 4, dtype=int)]),np.average(rawdata_AfterPed[np.arange(3500+2, 3500+52, 4, dtype=int)]),np.average(rawdata_AfterPed[np.arange(3500+3, 3500+52, 4, dtype=int)])])
    print("pedestal: %s"%(str(pedestal))) 
    rawdata_AfterPed = (rawdata_AfterPed.reshape((-1, 4)) - pedestal).flatten()
    # calculate residual
    residual_type1_AfterPed = np.sum(np.square(rawdata_AfterPed[np.where(mask_residual[0]==0)])) / len(np.where(mask_residual[0]==0))
    residual_type2_AfterPed = np.sum(np.square(rawdata_AfterPed[np.where(mask_residual[1]==0)])) / len(np.where(mask_residual[1]==0))
    residual_type2_cut5_AfterPed = np.sum(np.square(rawdata_AfterPed[np.where(mask_residual[2]==0)])) / len(np.where(mask_residual[2]==0))
    residual_type2_cut10_AfterPed = np.sum(np.square(rawdata_AfterPed[np.where(mask_residual[3]==0)])) / len(np.where(mask_residual[3]==0))
    residual_type2_cut50_AfterPed = np.sum(np.square(rawdata_AfterPed[np.where(mask_residual[4]==0)])) / len(np.where(mask_residual[4]==0))
    # correction after pedstal
    totalMuCorr=np.sum(rawdata_AfterPed[np.where(activeBXMask==1)])
    hfAfterPedTotalScale = totalMuCorr/totalMuUn if totalMuUn >0 else 0
    return [rawdata_AfterPed,hfAfterPedTotalScale],[ residual_type1_AfterPed, residual_type2_AfterPed, residual_type2_cut5_AfterPed, residual_type2_cut10_AfterPed, residual_type2_cut50_AfterPed]

def Makeplots(rawdata,minv, maxv, fracThreshold, dynamicThreshold, activeBXMask, rawdata_AfterGlow,rawdata_AfterPed, make_plots=False,HFSBR2=''):
    if make_plots:
            can=ROOT.TCanvas("can","",1200,1200)
            #if dooccupancy:
            if 2+2==5:
                if HFSBR2 !='':
                    can.Divide(2,2)
                else:
                    can.Divide(1,2)
            else:
                if HFSBR2 !='':
                    can.Divide(2,3)
                else:
                    can.Divide(1,3)
            if dooccupancy:
            #if 2+2==5:     
                rawTHist=ROOT.TH1F("rawTHist",";bunch crossing;Average Occupancy",3564,0,3564)
                rawTHistAfterG=ROOT.TH1F("rawTHistAfterG","set;bunch crossing;Average Occupancy",3564,0,3564)
                rawTHistAfterPed=ROOT.TH1F("rawTHistAfterPed","set;bunch crossing;Average Occupancy",3564,0,3564)
            else:
                rawTHist=ROOT.TH1F("rawTHist",";bunch crossing;Average Sum ET",3564,0,3564)
                rawTHistAfterG=ROOT.TH1F("rawTHistAfterG","set;bunch crossing;Average Sum ET",3564,0,3564)
                rawTHistAfterPed=ROOT.TH1F("rawTHistAfterPed","set;bunch crossing;Average Sum ET",3564,0,3564)
            ####################################################
            for ibx in range(MAX_NBX):
                rawTHist.Fill(ibx,rawdata[ibx])
            can.cd(1)
            rawTHist.SetMaximum(3)
            rawTHist.SetMinimum(0.0)
            rawTHist.SetMaximum(0.05)
            rawTHist.SetTitle("Raw Data")
            rawTHist.SetStats(00000)
            rawTHist.Draw("hist")
            line_minv = ROOT.TLine(0,minv,3564,minv)
            line_maxv = ROOT.TLine(0,maxv,3564,maxv)
            line_dynther = ROOT.TLine(0,minv + dynamicThreshold,3564,minv + dynamicThreshold)
            line_minv.SetLineColor(ROOT.kRed)
            line_maxv.SetLineColor(ROOT.kBlue)
            line_dynther.SetLineColor(ROOT.kGreen)
            line_minv.Draw("sames")
            line_maxv.Draw("sames")
            line_dynther.Draw("sames")
            ####################################################
            for ibx in range(MAX_NBX):
                rawTHistAfterG.Fill(ibx,rawdata_AfterGlow[ibx])
            if HFSBR2 !='':
                can.cd(3)
            else:
                can.cd(2)
            rawTHistAfterG.SetMaximum(3)
            rawTHistAfterG.SetTitle("+AfterGlow corrections")
            rawTHistAfterG.SetStats(0000000)
            rawTHistAfterG.Draw("hist")
            rawTHistAfterG.SetMaximum(0.05)
            rawTHistAfterG.SetMinimum(0.0)
            if dooccupancy:
            #if 2+2 == 5:    
                rawTHistAfterG.SetMaximum(0.01)
                rawTHistAfterG.SetMinimum(-0.002)
            ####################################################
            if not dooccupancy:
            #if not 2+2==5:     
                for ibx in range(MAX_NBX):
                    rawTHistAfterPed.Fill(ibx,rawdata_AfterPed[ibx])
                if HFSBR2 !='':
                    can.cd(5)
                else:
                    can.cd(3)
                rawTHistAfterPed.SetMaximum(3)
                rawTHistAfterPed.SetTitle("+AfterGlow + Pedestal corrections")
                rawTHistAfterPed.SetStats(0000000)
                rawTHistAfterPed.Draw("hist")
                rawTHistAfterPed.SetMaximum(0.01)
                rawTHistAfterPed.SetMinimum(-0.01)
            ####################################################
            if HFSBR2 !='':
                HFSBR2=np.array(HFSBR2)
                HFSBR2[0]=0
                rawdata2=np.copy(rawdata)
                for ibx in np.argwhere(activeBXMask==1):
                    rawdata2 -= rawdata2[ibx]*np.roll(HFSBR2, ibx)
                muafterglowPerBX2=np.copy(rawdata2)
                pedestal2 = np.array([np.average(rawdata2[np.arange(3500+0, 3500+52, 4, dtype=int)]),np.average(rawdata2[np.arange(3500+1, 3500+52, 4, dtype=int)]),np.average(rawdata2[np.arange(3500+2, 3500+52, 4, dtype=int)]),np.average(rawdata2[np.arange(3500+3, 3500+52, 4, dtype=int)])])
                rawdata2 = (rawdata2.reshape((-1, 4)) - pedestal2).flatten()
                if dooccupancy:
                    rawTHistAfterG2=ROOT.TH1F("rawTHistAfterG2","set;bunch crossing;Average Occupancy",3564,0,3564)
                    rawTHistAfterPed2=ROOT.TH1F("rawTHistAfterPed2","set;bunch crossing;Average Occupancy",3564,0,3564)
                else:
                    rawTHistAfterG2=ROOT.TH1F("rawTHistAfterG2","set;bunch crossing;Average Sum ET",3564,0,3564)
                    rawTHistAfterPed2=ROOT.TH1F("rawTHistAfterPed2","set;bunch crossing;Average Sum ET",3564,0,3564)
                can.cd(4)
                for ibx in range(MAX_NBX):
                    rawTHistAfterG2.Fill(ibx,muafterglowPerBX2[ibx])
                rawTHistAfterG2.SetMaximum(3)
                rawTHistAfterG2.SetTitle("+AfterGlow corrections")
                rawTHistAfterG2.SetStats(0000000)
                rawTHistAfterG2.Draw("hist")
                rawTHistAfterG2.SetMaximum(0.05)
                rawTHistAfterG2.SetMinimum(0.0)
                if dooccupancy:
                #if 2+2 == 5:   
                    rawTHistAfterG2.SetMaximum(0.01)
                    rawTHistAfterG2.SetMinimum(-0.002)
                #if not dooccupancy:
                if not 2 + 2 == 5: 
                    can.cd(6)
                    for ibx in range(MAX_NBX):
                        rawTHistAfterPed2.Fill(ibx,rawdata2[ibx])
                    rawTHistAfterPed2.SetMaximum(3)
                    rawTHistAfterPed2.SetTitle("+AfterGlow + Pedestal corrections")
                    rawTHistAfterPed2.SetStats(0000000)
                    rawTHistAfterPed2.Draw("hist")
                    rawTHistAfterPed2.SetMaximum(0.01)
                    rawTHistAfterPed2.SetMinimum(-0.01)
            #error on next line on purpose, to automatically close files and exit gracefully
            input()

def plt_sbr(HFSBR,HFSBR2=""):
    cc=ROOT.TCanvas()
    hh=ROOT.TH1F("hh","Original;;",len(HFSBR),0,len(HFSBR))
    for ibx in range(len(HFSBR)):
        hh.Fill(ibx,HFSBR[ibx])
    hh.SetLineColor(ROOT.kRed)
    hh.SetLineWidth(2)
    hh.Draw("HIST")
    hh.SetStats(0000000)
    if HFSBR2!="":
        hh2=ROOT.TH1F("hh2","New;;",len(HFSBR2),0,len(HFSBR2))
        for ibx in range(len(HFSBR2)):
            hh2.Fill(ibx,HFSBR2[ibx])
        hh2.SetLineWidth(2)
        hh2.SetLineColor(ROOT.kBlue)
        hh2.Draw("HIST same")
        hh2.SetStats(0000000)
    cc.BuildLegend()
    input()

def compute_residuals(ix, iy, this_res_, params, xy=(-1,-1), makeplot=False):

    if xy[0] > -1: params[xy[0]] = ix
    if xy[1] > -1: params[xy[1]] = iy
    func_p0 = params[0] #90.49 #93.04
    func_p1 = params[1] #4.750e-4 #4.478e-4 #4.067e-4
    #func_p2 is the one we choose to be dependent to make sure func matches func2 at boundary
    func_p3 = params[2] #91.05 
    func_p4 = params[3] #54.78 #26.95
    func_p5 = params[4] #1.417e-5 #5.334e-5
    func2_param0 = params[5] #1.080e-4 # 1.147e-4 
    func2_param1 = params[6] #685.1 #793.1 
    # Best for ET method
    ##############
    if not dooccupancy:
        func=ROOT.TF1("func","exp((-x)/[0])*[1] + [2] + exp(-(x-[3])*(x-[3])/[4]/[4])*[5] ",0,5000)
        #func_p0 = params[0] #90.49 #93.04
        #func_p1 = params[1] #4.750e-4 #4.478e-4 #4.067e-4
        ##func_p2 is the one we choose to be dependent to make sure func matches func2 at boundary
        #func_p3 = params[2] #91.05 
        #func_p4 = params[3] #54.78 #26.95
        #func_p5 = params[4] #1.417e-5 #5.334e-5
        #func2_param0 = params[5] #1.080e-4 # 1.147e-4 
        #func2_param1 = params[6] #685.1 #793.1 
        terma = func2_param0*exp(-180./func2_param1)
        termb = -1* func_p1*exp(-180./func_p0)
        termc =   -1* func_p5*exp(-(180.-func_p3)**2 / func_p4**2)
        func_param2 = terma + termb + termc
        #print("func2_param0: %f ; func2_param1: %f ; terma: %f ; termb: %f ; termc: %f ; func_param2: %f"%(func2_param0, func2_param1, terma, termb, termc, func_param2)) 
        ##func.SetParameters(67.5,4.58e-4,6e-05, 85, 17,7e-5)
        ##func.SetParameters(80.43, 4.019e-4, 6.055e-5, 88.814, 20.80, 5.760e-5)
        #func.SetParameters(79.29, 4.067e-4, 6.027e-5, 105.87, 26.95, 5.334e-5)
        #func.SetParameters(79.29, 4.067e-4, 6.027e-5, 105.87, ix, 1.3e-10)
        func.SetParameters(func_p0, func_p1, func_param2, func_p3, func_p4, func_p5)
        #????
        #func.SetParameters(72.2, 4.98e-4, 6.29e-5, 67.4, 16.3, 7.08e-5)
        func2=ROOT.TF1("func2","exp((-x)/[1])*[0]",0,5000)
        #func2.SetParameters(764.12490, 0.00010825920)
        #func2.SetParameters(837.552, 9.0665e-5)
        #func2.SetParameters(835.080, 9.022e-5)
        #func2_param1 = iy #8.5853e-5
        #func2_param0 = iy 
        #func2_param0 = (func.GetParameter(0)*exp(-180/func.GetParameter(1)) + func.GetParameter(2) + 
        #    func.GetParameter(5)*exp(-(180 - func.GetParameter(3))**2 / func.GetParameter(4)**2) ) * exp(180 / func2_param1) 
        #terma = func.GetParameter(0)*exp(-180.0/func.GetParameter(1) + 180.0/func2_param1)
        #termb = func.GetParameter(2)*exp(180.0/func2_param1)
        #termc = func.GetParameter(5)*exp(-(180 - func.GetParameter(3))**2 / func.GetParameter(4)**2 + 180.0/func2_param1)
        #print("func2_param1: " + str(func2_param1)) 
        #func2_param0 = terma + termb + termc
        #func2.SetParameters(935.80, 8.5853e-5) 
        #func2_param1 = -180. / log( func.Eval(180) / func2_param0 ) 
        #print("func2_param0: %f, func(180): %f, func2_param1: %f"%(func2_param0, func.Eval(180), func2_param1)) 
        func2.SetParameters(func2_param0, func2_param1) 
        #print("func2 @x=180:%f ; 200:%f ; 500:%f ; 1000:%f ; 2000:%f ; 3000: %f"%(func2.Eval(180), func2.Eval(200), func2.Eval(500), func2.Eval(1000), func2.Eval(2000), func2.Eval(3000))) 
        for i in range(2,180):
            HFSBR_new[i]= func.Eval(i)
        for i in range(180,len(HFSBR_new)):
            HFSBR_new[i]= func2.Eval(i)
        #HFSBR_new[1]=0.02847
        #HFSBR_new[1]=.03208
        HFSBR_new[1]=params[7] #.0323833
        if write_file:
            #file = open("HFSBR_ET_22v11.txt", "w+")
            file = open(args.outwrite, "w+")
            for ibxwrite in HFSBR_new:
                file.write('%0.10f'%ibxwrite)
                if ibxwrite!=HFSBR_new[-1]: file.write(', ')
            file.close()
            exit()
    ##############
    # Best for OC method
    if dooccupancy:
        #func=ROOT.TF1("func","exp((-x)/[0])*[1] + [2] + exp(-(x-[3])*(x-[3])/[4]/[4])*[5] + exp(-(x-[6])*(x-[6])/[7]/[7])*[8]",0,5000)
        func=ROOT.TF1("func","exp((-x)/[0])*[1] + [2] + exp(-(x-[3])*(x-[3])/[4]/[4])*[5]",0,5000)
        #func_p0 = ix #37.85
        #func_p1 = 5.260e-4
        ##func_p2 is the one we choose to be dependent to make sure func matches func2 at boundary
        #func_p3 = 98.85
        #func_p4 = 118.9
        #func_p5 = iy #1.676e-4
        #func2_param0 = 1.37e-4 #ix 
        #func2_param1 = 829.2 #iy 
        terma = func2_param0*exp(-180./func2_param1)
        termb = -1* func_p1*exp(-180./func_p0)
        termc =   -1* func_p5*exp(-(180.-func_p3)**2 / func_p4**2)
        func_param2 = terma + termb + termc
        #func_param2 = func2_param0*exp(-180./func2_param1) - func_p1*exp(-180./func_p0) - func_p5*exp(-(180.-func_p3)**2 / func_p4**2)
        #func=ROOT.TF1("func","exp((-x)/[0])*[1] + [2]",0,5000)
        #func.SetParameters(80, 5.1e-4, 3.7e-05, 98.385000, 200 , 4e-05,   85.264350, 24, 2.2e-05)
        #func.SetParameters(35.22, 5.192e-4, 3.999e-5, 98.85, 161.8 , 1.676e-4,   0, 0, 0)
        #func.SetParameters(37.85, 5.260e-4, 3.906e-5, 98.85, 118.9, 1.676e-4,   132.37, 5.635, 1.052e-4)
        #func.SetParameters(37.85, 5.260e-4, 3.906e-5, 98.85, 118.9, 1.676e-4, 136.2, 5.163, 1.052e-4)
        #func.SetParameters(37.85, 5.260e-4, func_param2, 98.85, 118.9, 1.676e-4)
        func.SetParameters(func_p0, func_p1, func_param2, func_p3, func_p4, func_p5)
        func2=ROOT.TF1("func2","exp((-x)/[1])*[0]",0,5000)
        #func2_param0 = (func.GetParameter(0)*exp(-180/func.GetParameter(1)) + func.GetParameter(2) + 
        #    func.GetParameter(5)*exp(-(180 - func.GetParameter(3))**2 / func.GetParameter(4)**2) ) * exp(180 / func2_param1) 
        #func2.SetParameters(2735, 8.851e-5)
        func2.SetParameters(func2_param0, func2_param1)
        for i in range(2,190):
            HFSBR_new[i]= func.Eval(i)
        for i in range(190,len(HFSBR_new)):
            HFSBR_new[i]= func2.Eval(i)
        #HFSBR_new[1]= 0.01102
        HFSBR_new[1]=params[7] # 0.0190175
        if write_file:
            file = open("HFSBR_OC_22v11.txt", "w+")
            for ibxwrite in HFSBR_new:
                file.write('%0.10f'%ibxwrite)
                if ibxwrite!=HFSBR_new[-1]: file.write(', ')
            file.close()
            exit()
    #plt_sbr(HFSBR,HFSBR_new)
    result_from_dynamic=MakeDynamicBunchMask(thisLNHists['mu']['total'])
    #print("rawdata!!!: ")
    #print(thisLNHists['mu']['total']) 
#[rawdata, minv, maxv, fracThreshold, dynamicThreshold, activeBXMask, NActiveBX]
    #coreq one below!!
    result_from_afterglow,mask_residual,residual_AfterGlow=ComputeAfterglow(result_from_dynamic[0], result_from_dynamic[5], HFSBR_new)
    #HFSBR_zero = np.zeros(len(HFSBR)) 
    #result_from_afterglow,mask_residual,residual_AfterGlow=ComputeAfterglow(result_from_dynamic[0], result_from_dynamic[5], HFSBR_zero)
    #print("result from afterglow!!: ")
    #print(result_from_afterglow)
#[ rawdata_Afterglow,hfAfterGlowTotalScale, totalMuUn], [mask_residual_type1, mask_residual_type2, mask_residual_type2_cut5, mask_residual_type2_cut10, mask_residual_type2_cut50], [residual_type1_AfterGlow, residual_type2_AfterGlow, residual_type2_cut5_AfterGlow, residual_type2_cut10_AfterGlow, residual_type2_cut50_AfterGlow]
    result_from_pedsub,residual_AfterPed=SubtractPedestal(result_from_afterglow[0], mask_residual, result_from_dynamic[5], result_from_afterglow[2])
    #print("residual_AfterPed: %f"%(residual_AfterPed[1])) 
    #print("this_res_ before modifying: %f"%(this_res_[str(ix)+str(iy)]['type2_afterPed'])) 
#[rawdata_AfterPed,hfAfterPedTotalScale],[ residual_type1_AfterPed, residual_type2_AfterPed, residual_type2_cut5_AfterPed, residual_type2_cut10_AfterPed, residual_type2_cut50_AfterPed]
    if makeplot:
        Makeplots(result_from_dynamic[0], result_from_dynamic[1],result_from_dynamic[2],result_from_dynamic[3],result_from_dynamic[4], result_from_dynamic[5], result_from_afterglow[0], result_from_pedsub[0], True,HFSBR)
    this_res_[str(ix)+str(iy)]['type1_afterGlow']          +=residual_AfterGlow[0]
    this_res_[str(ix)+str(iy)]['type2_afterGlow']          +=residual_AfterGlow[1]
    this_res_[str(ix)+str(iy)]['type2_cut5_afterGlow']     +=residual_AfterGlow[2]
    this_res_[str(ix)+str(iy)]['type2_cut10_afterGlow']    +=residual_AfterGlow[3]
    this_res_[str(ix)+str(iy)]['type2_cut50_afterGlow']    +=residual_AfterGlow[4]
    this_res_[str(ix)+str(iy)]['lumi_correction_afterGlow']+=result_from_afterglow[1]
    this_res_[str(ix)+str(iy)]['type1_afterPed']           +=residual_AfterPed[0]
    this_res_[str(ix)+str(iy)]['type2_afterPed']           +=residual_AfterPed[1]
    #print("this_res_ after modifying: %f"%(this_res_[str(ix)+str(iy)]['type2_afterPed'])) 
    this_res_[str(ix)+str(iy)]['type2_cut5_afterPed']      +=residual_AfterPed[2]
    this_res_[str(ix)+str(iy)]['type2_cut10_afterPed']     +=residual_AfterPed[3]
    this_res_[str(ix)+str(iy)]['type2_cut50_afterPed']     +=residual_AfterPed[4]
    this_res_[str(ix)+str(iy)]['lumi_correction_afterPed'] +=result_from_pedsub[1]

    #print("at end of compute_residuals function, this_res_[%f %f]['type2_afterPed']: %f"%(ix, iy, this_res_[str(ix)+str(iy)]['type2_afterPed'])) 
    dataQueue.put((ix, iy, this_res_[str(ix)+str(iy)])) 
    return #ix, iy, this_res_

#multiprocessing function to call 'compute_residuals'
def compute_residuals_multi(package):
    #print("NOW calling compute_residuals on %s"%(str(package))) 
    return compute_residuals(*package)

def optimize(xval, yval, unct, params, xedge=0, yedge=0, xy=(-1,-1)):
    start = time.time()
    nset=0
    nset_perfile=0
    tot_res=0 
    global tot_res_
   # tot_res_['type1_afterGlow']={}
   # tot_res_['type2_afterGlow']={}
   # tot_res_['type2_cut5_afterGlow']={}
   # tot_res_['type2_cut10_afterGlow']={}
   # tot_res_['type2_cut50_afterGlow']={}
   # tot_res_['lumi_correction_afterGlow']={}
   # tot_res_['type1_afterPed']={}
   # tot_res_['type2_afterPed']={}
   # tot_res_['type2_cut5_afterPed']={}
   # tot_res_['type2_cut10_afterPed']={}
   # tot_res_['type2_cut50_afterPed']={}
   # tot_res_['lumi_correction_afterPed']={}

    #previous parameter values: 80.43, 4.019e-4, 6.055e-5, 88.814, 20.80, 5.760e-5
    #  (for ET)                 837.552, 9.0665e-5
    # Now here for OC: 80, 5.1e-4, 3.7e-05, 98.385000, 200, 4e-05, 85.264350, 24, 2.2e-05
    #                           2152.6, 1.1e-4 
    #xmin_=0.01320*0.5
    #xmax_=0.01320*1.5
    #range of percents to alter the values
    #98.28, 23.99 ; 835.080, 9.022e-5
    # 100.566, 25.33
    #xedge/yedge: if the minimum occurred at the edge of the range
    #edges: -1 for low edge, +1 for high edge, 0 for not on the edge
    #xedge = 0
    #yedge = -1
    #unct=0.1
    ##2735, 8.851e-5
    #central x value (will vary up and down by unct*xval)
    #xval=1.172e-4
    xmin_=xval*(1.0-unct + xedge*unct) 
    xmax_=xval*(1.0+unct + xedge*unct) #67.5*1.1
    #how many bins of different x values to try
    XN=9
    if xy[0] == -1: XN = 1
    #ymin_=9.2020320e-05*1.18*0.5
    #ymax_=9.2020320e-05*1.18*3.5
    #central y value (will vary up and down by unct*yval)
    #yval=1387 
    ymin_=yval*(1.0-unct + yedge*unct)
    ymax_=yval*(1.0+unct + yedge*unct)
    #how many bins of different y values to try
    YN=9
    if xy[1] == -1: YN = 1
    # after glow
    residualhist_type1_afterGlow=ROOT.TH2F("residualhist_type1_afterGlow",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_afterGlow=ROOT.TH2F("residualhist_type2_afterGlow",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_cut5_afterGlow=ROOT.TH2F("residualhist_type2_cut5_afterGlow",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_cut10_afterGlow=ROOT.TH2F("residualhist_type2_cut10_afterGlow",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_cut50_afterGlow=ROOT.TH2F("residualhist_type2_cut50_afterGlow",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    lumicorrectionhist_afterGlow=ROOT.TH2F("lumicorrectionhist_afterGlow",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type1_afterGlow_1D=ROOT.TH1F("residualhist_type1_afterGlow_1D",";xx;yy",XN,xmin_,xmax_)
    # after ped
    residualhist_type1_afterPed=ROOT.TH2F("residualhist_type1_afterPed",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_afterPed=ROOT.TH2F("residualhist_type2_afterPed",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_cut5_afterPed=ROOT.TH2F("residualhist_type2_cut5_afterPed",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_cut10_afterPed=ROOT.TH2F("residualhist_type2_cut10_afterPed",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type2_cut50_afterPed=ROOT.TH2F("residualhist_type2_cut50_afterPed",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    lumicorrectionhist_afterPed=ROOT.TH2F("lumicorrectionhist_afterPed",";xx;yy",XN,xmin_,xmax_,YN,ymin_,ymax_)
    residualhist_type1_afterPed_1D=ROOT.TH1F("residualhist_type1_afterPed_1D",";xx;yy",XN,xmin_,xmax_)
    # initialize
    d = mp.Manager().dict()
    for iix in range(residualhist_type1_afterGlow.GetNbinsX()):
        for iiy in range(residualhist_type1_afterGlow.GetNbinsY()):
            ix = residualhist_type1_afterGlow.GetXaxis().GetBinCenter(iix+1)
            iy = residualhist_type1_afterGlow.GetYaxis().GetBinCenter(iiy+1)
            #print("ix=%f, iy=%f"%(ix, iy)) 
            tot_res_[str(ix)+str(iy)]={} 
            for thing in ['type1_afterGlow', 'type2_afterGlow', 'type2_cut5_afterGlow', 'type2_cut10_afterGlow', 'type2_cut50_afterGlow', 'lumi_correction_afterGlow', 'type1_afterPed', 'type2_afterPed', 'type2_cut5_afterPed', 'type2_cut10_afterPed', 'type2_cut50_afterPed', 'lumi_correction_afterPed']:
                tot_res_[str(ix)+str(iy)][thing] = 0
                
 
    #keep track of the minimum residual for any of the parameter values tested
    min_res = 999999
    x_min = -999999
    y_min = -999999
    for ifile in range(0,len(hists)):
        # Start reading HISTS
        nset_perfile=0
        NOT_END=[True,True] # length of hfCMS_VALID and hfCMS_CMS1 are not the same
        icolstart=[0,0]
        lastLN=-1
        thisLN=-1
        while (NOT_END[0] and NOT_END[1]):
            for ihist in [0,1]:
                hist = hists[ifile][ihist]
                for icols in range(icolstart[ihist],len(hist.cols.runnum)):
                    thisLN=hist.cols.runnum[icols]*1e7+hist.cols.lsnum[icols]*1e2+hist.cols.nbnum[icols]
                    boardKey=(hist.cols.datasourceid[icols],hist.cols.channelid[icols])
                    if lastLN==-1:
                        lastLN=thisLN
                    if lastLN!=thisLN:
                        icolstart[ihist]=icols
                        if ihist==1: 
                            nset+=1
                            nset_perfile+=1
                            lastLN=thisLN
                            #print(icolstart)
                            #print('hit new ln ', lastLN, thisLN, nset)
                            global thisLNHists
                            thisLNHists['hfCMS_VALID']['total']=np.zeros(3564)
                            if dooccupancy:
                                thisLNHists['hfCMS1']['total']= np.zeros(3564)
                            else:
                                thisLNHists['hfCMS_ET']['total']= np.zeros(3564)
                            thisLNHists['mu']['total']= np.zeros(3564)
                            nBoards=0
                            for boardFound in thisLNHists['hfCMS_VALID'].keys():
                                if boardFound is 'total':
                                    continue
                                if dooccupancy:
                                    if thisLNHists['hfCMS1'].has_key(boardFound):
                                        nBoards=nBoards+1
                                        thisLNHists['mu'][boardFound]= np.zeros(3564)
                                        thisLNHists['hfCMS_VALID']['total']+=thisLNHists['hfCMS_VALID'][boardFound]
                                        thisLNHists['hfCMS1']['total']+=thisLNHists['hfCMS1'][boardFound]
                                        boardComp = np.greater(thisLNHists['hfCMS1'][boardFound], thisLNHists['hfCMS_VALID'][boardFound])
                                        maxval = np.max(thisLNHists['hfCMS_VALID'][boardFound]) 
                                        max1 = np.max(thisLNHists['hfCMS1'][boardFound]) 
                                        print("max value for thiLNHists[valid]: %f ; [cms1]: %f"%(maxval, max1)) 
                                        print("board comparison: " + str(boardComp)) 
                                        hiCMS1 = thisLNHists['hfCMS1'][boardFound][boardComp] 
                                        loCMSV = thisLNHists['hfCMS_VALID'][boardFound][boardComp]
                                        print("CMS1 too high: " + str(hiCMS1))
                                        print("or is it CMS_VALID too low??: " + str(loCMSV))
                                        thisLNHists['mu'][boardFound]=np.divide(thisLNHists['hfCMS1'][boardFound],thisLNHists['hfCMS_VALID'][boardFound],out=np.zeros(thisLNHists['hfCMS1'][boardFound].shape, dtype=float), where=thisLNHists['hfCMS_VALID'][boardFound]>0)
                                else:
                                    if thisLNHists['hfCMS_ET'].has_key(boardFound):
                                        nBoards=nBoards+1
                                        thisLNHists['mu'][boardFound]= np.zeros(3564)
                                        thisLNHists['hfCMS_VALID']['total']+=thisLNHists['hfCMS_VALID'][boardFound]
                                        thisLNHists['hfCMS_ET']['total']+=thisLNHists['hfCMS_ET'][boardFound]
                                        thisLNHists['mu'][boardFound]=np.divide(thisLNHists['hfCMS_ET'][boardFound],thisLNHists['hfCMS_VALID'][boardFound],out=np.zeros(thisLNHists['hfCMS_ET'][boardFound].shape, dtype=float), where=thisLNHists['hfCMS_VALID'][boardFound]>0)
                                        boardComp = np.greater(thisLNHists['hfCMS_ET'][boardFound], thisLNHists['hfCMS_VALID'][boardFound])
                                        maxval = np.max(thisLNHists['hfCMS_VALID'][boardFound]) 
                                        max1 = np.max(thisLNHists['hfCMS_ET'][boardFound]) 
                                        print("max value for thiLNHists[valid]: %f ; [cmsET]: %f"%(maxval, max1)) 
                                        print("board comparison: " + str(boardComp)) 
                                        hiCMS1 = thisLNHists['hfCMS_ET'][boardFound][boardComp] 
                                        loCMSV = thisLNHists['hfCMS_VALID'][boardFound][boardComp]
                                        print("CMSET too high: " + str(hiCMS1))
                                        print("or is it CMS_VALID too low??: " + str(loCMSV))
                            #print("nBoards: %d"%nBoards) 
                            #print("thisLNHists[oc]: " + str(thisLNHists['hfCMS1']['total']))
                            #print("thisLNHists[valid]: " + str(thisLNHists['hfCMS_VALID']['total']))
                            #comparison=np.greater(thisLNHists['hfCMS1']['total'], thisLNHists['hfCMS_VALID']['total'])
                            #cms1Hi = thisLNHists['hfCMS1']['total'][comparison]
                            #valLow = thisLNHists['hfCMS_VALID']['total'][comparison]
                            #print("All elements with CMS1 > valid: \nOC: " + str(cms1Hi) + "\nValid: " + str(valLow))
                            if dooccupancy:
                                thisLNHists['mu']['total']=np.divide(thisLNHists['hfCMS1']['total'],thisLNHists['hfCMS_VALID']['total'],out=np.zeros(thisLNHists['hfCMS1']['total'].shape, dtype=float), where=thisLNHists['hfCMS_VALID']['total']>0)
                                #print("thisLNHists[mu][total] greater than 1:")
                                #selected = thisLNHists['mu']['total'] > 1
                                #print(thisLNHists['mu']['total'][selected])
                                thisLNHists['mu']['total']=-np.log(1-thisLNHists['mu']['total'])
                            else:
                                thisLNHists['mu']['total']=np.divide(thisLNHists['hfCMS_ET']['total'],thisLNHists['hfCMS_VALID']['total'],out=np.zeros(thisLNHists['hfCMS_ET']['total'].shape, dtype=float), where=thisLNHists['hfCMS_VALID']['total']>0)
                            thisLNHists['mu']['total'][3481:3500] = 0
                            
                            #put together the package for multiprocessing
                            package = []
                            makeplot = args.processingcores == 1 and args.makeplot
                            #if args.processingcores > 1:
                            #    for iix in range(residualhist_type1_afterGlow.GetNbinsX()): 
                            #        for iiy in range(residualhist_type1_afterGlow.GetNbinsY()):
                            #            ix = residualhist_type1_afterGlow.GetXaxis().GetBinCenter(iix+1)
                            #            iy = residualhist_type1_afterGlow.GetYaxis().GetBinCenter(iiy+1) 
                            #            d[str(ix)+str(iy)] = tot_res_[str(ix)+str(iy)] 
                            #else:
                            d = tot_res_
                            out_queue = mp.Queue()
                            #print("BEFORE calling pool, d: %s"%(str(d))) 
                            for iix in range(residualhist_type1_afterGlow.GetNbinsX()): 
                                for iiy in range(residualhist_type1_afterGlow.GetNbinsY()):
                                    ix = residualhist_type1_afterGlow.GetXaxis().GetBinCenter(iix+1)
                                    iy = residualhist_type1_afterGlow.GetYaxis().GetBinCenter(iiy+1) 
                                    package.append( (ix, iy, d, params, xy, makeplot) )

                            print("Now working on ifile: %d"%(ifile))  
                            if args.processingcores > 1:
                                pool  = mp.Pool(args.processingcores)
                                workers = pool.map(compute_residuals_multi, package)
                                #print("out q: %s"%(str(pool._outqueue)))
                                #print("q size: %d"%(pool._outqueue.qsize())) 
                                #while pool._outqueue.qsize() > 0:
                                pool.close()
                                print("waiting for pool to join....(ifile %d)"%(ifile)) 
                                pool.join()
                                print("pool joined!") 
                                if dataQueue.empty(): sys.exit("error: queueueueueu empty!!!!")
                                while not dataQueue.empty():
                                    res = dataQueue.get() #pool._outqueue.get()
                                    #print("res!!!!!!!!" + str(res))
                                    tot_res_[str(res[0])+str(res[1])] = res[2] 
                                    #print("ix: %f, iy: %f"%(res[0], res[1])) 
                                #for iix in range(residualhist_type1_afterGlow.GetNbinsX()): 
                                #    for iiy in range(residualhist_type1_afterGlow.GetNbinsY()):
                                #        ix = residualhist_type1_afterGlow.GetXaxis().GetBinCenter(iix+1)
                                #        iy = residualhist_type1_afterGlow.GetYaxis().GetBinCenter(iiy+1) 
                                #        #print("after one compute_residuals function calls, tot_res_[%f %f]['type2_afterPed']: %f"%(ix, iy, tot_res_[str(ix)+str(iy)]['type2_afterPed'])) 
                            else:
                                for pack in package:
                                    compute_residuals(*pack)
                                

                            #Reset Arrays
                            thisLNHists={}
                            thisLNHists['hfCMS_VALID']={}
                            thisLNHists['hfCMS1']={}
                            thisLNHists['hfCMS_ET']={}
                            thisLNHists['mu']={}
                        break
                    if icols == len(hist.cols.runnum)-1: NOT_END[ihist]=False
                    thisLNHists[hist.name][boardKey] = np.zeros(3564)
                    thisLNHists[hist.name][boardKey] = np.array(hist.cols.data[icols])
            #if nset_perfile==50: break
    for iix in range(residualhist_type1_afterGlow.GetNbinsX()):
        for iiy in range(residualhist_type1_afterGlow.GetNbinsY()):
            ix = residualhist_type1_afterGlow.GetXaxis().GetBinCenter(iix+1)
            iy = residualhist_type1_afterGlow.GetYaxis().GetBinCenter(iiy+1)
            #print("after completing compute_residuals function calls, tot_res_[%f %f]['type2_afterPed']: %f"%(ix, iy, tot_res_[str(ix)+str(iy)]['type2_afterPed']))
            #print("iix=%d ; iiy=%d ix=%f ; iy=%f ; tot_res_ = %f"%(iix, iiy, ix, iy, tot_res_[str(ix)+str(iy)]['type1_afterGlow']/nset)) 
            residualhist_type1_afterGlow.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type1_afterGlow']/nset)
            residualhist_type1_afterGlow_1D.Fill(ix,tot_res_[str(ix)+str(iy)]['type1_afterGlow']/nset)
            residualhist_type2_afterGlow.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_afterGlow']/nset)
            residualhist_type2_cut5_afterGlow.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_cut5_afterGlow']/nset)
            residualhist_type2_cut10_afterGlow.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_cut10_afterGlow']/nset)
            residualhist_type2_cut50_afterGlow.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_cut50_afterGlow']/nset)
            lumicorrectionhist_afterGlow.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['lumi_correction_afterGlow']/nset)
            residualhist_type1_afterPed.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type1_afterPed']/nset)
            residualhist_type1_afterPed_1D.Fill(ix,tot_res_[str(ix)+str(iy)]['type1_afterPed']/nset)
            residualhist_type2_afterPed.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_afterPed']/nset)
            residualhist_type2_cut5_afterPed.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_cut5_afterPed']/nset)
            residualhist_type2_cut10_afterPed.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_cut10_afterPed']/nset)
            residualhist_type2_cut50_afterPed.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['type2_cut50_afterPed']/nset)
            lumicorrectionhist_afterPed.Fill(ix,iy,tot_res_[str(ix)+str(iy)]['lumi_correction_afterPed']/nset)

            #see if this is the new min
            typename = 'type2_afterPed'
            if xy[0] == 7: typename = 'type1_afterPed'
            my_res = tot_res_[str(ix)+str(iy)][typename]/nset
            if my_res < min_res:
                min_res = my_res
                x_min = ix
                y_min = iy
                if iix == residualhist_type1_afterGlow.GetNbinsX()-1:
                    xedge = 1
                elif iix == 0:
                    xedge = -1
                else:
                    xedge = 0
                if iiy == residualhist_type1_afterGlow.GetNbinsY()-1:
                    yedge = 1
                elif iiy == 0:
                    yedge = -1
                else:
                    yedge = 0
            elif min_res > 999:
                print("ERROR: how is the residual not lower????")
                print("residual_AfterPed: %f ; residual_AfterGlow: %f"%(my_res, tot_res_[str(ix)+str(iy)]['type2_afterGlow']/nset)) 
                exit()


    f=ROOT.TFile("testv6_"+args.out+".root","RECREATE")
    residualhist_type1_afterGlow.Write()
    residualhist_type2_afterGlow.Write()
    residualhist_type2_cut5_afterGlow.Write()
    residualhist_type2_cut10_afterGlow.Write()
    residualhist_type2_cut50_afterGlow.Write()
    lumicorrectionhist_afterGlow.Write()
    residualhist_type1_afterGlow_1D.Write()
    residualhist_type1_afterPed.Write()
    residualhist_type2_afterPed.Write()
    residualhist_type2_cut5_afterPed.Write()
    residualhist_type2_cut10_afterPed.Write()
    residualhist_type2_cut50_afterPed.Write()
    lumicorrectionhist_afterPed.Write()
    residualhist_type1_afterPed_1D.Write()
    f.Close()
    print("MINIMUM RESIDUAL: %f ; FOUND AT x = %f ; y = %f"%(min_res, x_min, y_min))
    print("original xval: %f, original yval: %f, xedge: %d, yedge: %d"%(xval, yval, xedge, yedge)) 
    runtime = time.time() - start
    print("%d CORES; Total time to run: %f minutes"%(args.processingcores, runtime/60.0))
    #exit()
    return min_res, x_min, y_min, xedge, yedge
        
        
params = [0.0 for i in range(8)]
def read_params(paramfile):
    pf = open(paramfile, "r")
    i = 0
    for line in pf:
        params[i] = float(line)
        i += 1
    pf.close()

read_params(args.paramfile)

#p0, p1 are the parameter numbers (0 to 7)
def opt_params(p0, p1):
#xy is the tuple of which parameters to modify the value of
    xy = (p0, p1)
    #if args.xparam > -1: xy = (args.xparam, xy[1])
    #if args.yparam > -1: xy = (xy[0], args.yparam)

    xval = params[xy[0]] #65.98 #1.25e-4
    yval = params[xy[1]] #1.085e-4 #1121
    unct = 0.1 #0.2
    xedge = 0
    yedge = 0
    min_res, xmin, ymin, xedge, yedge = optimize(xval, yval, unct, params, xedge, yedge, xy)

    #done should be true if the bin with minimum residual is not much better than the previous one
    done = False
    #cutoff is 1e-9 for type1, 1e-7 for type2
    cutoff = 1e-7
    if xy[0] == 7: cutoff = 1e-9
    prev_min = min_res
    while not done:
        print("doing next optimization!")
        prev_min = min_res
        if xedge == 0 and yedge == 0: unct/= 2
        prev_xedge = xedge
        prev_yedge = yedge
        min_res, xmin, ymin, xedge, yedge = optimize(xmin, ymin, unct, params, xedge, yedge, xy)
        if xy[1] == -1: yedge = 0
        done = xedge == 0 and yedge == 0 and abs(min_res - prev_min) < cutoff
        if xedge*prev_xedge == -1: xedge = 0
        if yedge*prev_yedge == -1: yedge = 0
    tot_time = time.time() - tot_start
    print("All done optimizing!! Total time: %f minutes"%(tot_time/60.0)) 
    print("Optimal values: param %d: %f ; param %d: %f. Residual: %f"%(xy[0], xmin, xy[1], ymin, min_res)) 
    return xmin, ymin

p0 = -1
p1 = -1
if not args.write:
    if args.xparam > -1: p0 = args.xparam
    if args.yparam > -1: p1 = args.yparam
#if no params are specified then optimize all params!
if p0 == -1 and p1 == -1:
    for p0 in range(7):
        for p1 in range(p0+1, 7):
            optP0, optP1 = opt_params(p0, p1)
            #update params and write these new params to the paramfile.
            params[p0] = optP0
            params[p1] = optP1
            pf = open(args.paramfile, "w")
            for par in params:
                pf.write(str(par) + "\n")
            pf.close()
    print("All parameters optimized!!! Total total total time fr fr fr ong: %f hours"%((time.time()-tot_start)/3600)) 
    print(params)
else:
    opt_params(p0, p1)

exit()
