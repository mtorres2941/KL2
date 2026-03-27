################################################################################################################
################################################################################################################

# import sys
# sys.path.insert(0, '../UsefulFunctions')

# from funcs_unit_conversion import hi2, hi3, lo2
# from funcs_unit_conversion import m2_ft2, m3_ft3, m3_yd3, psi_mpa

# from funcs_unit_conversion import area_units, m2, ft2
# from funcs_unit_conversion import densityarea_units, kg_m2, kg_yd2, psf, kg_dm2, t_m2, g_m2, oz_yd2
# from funcs_unit_conversion import density_units, kg_m3, kg_yd3, lb_ft3, kg_dm3, t_m3, g_m3, kg_l
# from funcs_unit_conversion import emission_units, lbco2e, tco2e, kgco2e
# from funcs_unit_conversion import energyemission_units, lbco2mwh, tco2mwh, kgco2mwh 
# from funcs_unit_conversion import length_units, m1, ft, cm, mm, inch, km
# from funcs_unit_conversion import pressure_units, psi, ksi, mpa, nmm2
# from funcs_unit_conversion import therm_units, uval, rval, rsival
# from funcs_unit_conversion import time_units, decade, year, day, hour, minute, second
# from funcs_unit_conversion import volume_units, ft3, m3, yd3
# from funcs_unit_conversion import weight_units, kgs, gs, lbs, tons, tonnes

# from funcs_unit_conversion import area2m2, density2kgm2, density2kgm3, emission2kgco2e, emission2kgmwh, length2in, pressure2psi, therm2rval, time2year, vol2m3, weight2kgs
# from funcs_unit_conversion import dict_unitconv, dict_unittype, consistent_units, str2valunit

################################################################################################################
################################################################################################################
#Initialize variables
import numpy as np
import pandas as pd

#prefixes and suffixes
hi2 = r'$\mathrm{\mathsf{   ^2   }}$'
hi3 = r'$\mathrm{\mathsf{   ^3   }}$'
lo2 = r'$\mathrm{\mathsf{   _2   }}$'

#conversion factors
m2_ft2 = 0.092903
m3_ft3 = 0.0283168
m3_yd3 = 0.764555
psi_mpa = 145.038


# area2m2          area_units
# density2kgm2     densityarea_units
# density2kgm3     density_units
# emission2kgco2e  emission_units
# emission2kgmwh   energyemission_units
# length2in        length_units
# pressure2psi     pressure_units
# therm2rval       therm_units
# time2year        time_units
# vol2m3           volume_units
# weight2kgs       weight_units

# str2valunit
# consistent_units

# dict_unitconv
# dict_unittype


##################################### area2m2 ###########################################################################
##################################### area2m2 ###########################################################################

m2 = ['m2', 'm²', 'm^2', 'm**2']
ft2 = ['ft2', 'sqft', 'sf', 'ft²', 'ft^2']
area_units = m2 + ft2


def area2m2(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in m2)
    """
    if unit in m2:
        pass
    
    elif unit in ft2:
        qty *= m2_ft2
        
    else:
        qty = None
        if prnt == 'y':
            print(f'{unit} is not being accounted for in the area2m2 function')
        
    return qty



################################## density2kgm2 ##############################################################################
################################## density2kgm2 ##############################################################################

kg_m2 = ['kg/m2', 'kg/m^2']
kg_yd2 = ['kg/yd2', 'kg/yd^2']
psf = ['lbs/ft2', 'lb/ft2', 'psf']
kg_dm2 = ['kg/dm2']   
t_m2 = ['t/m2', 't/m^2']
g_m2 = ['g/m2', 'g/m^2']
oz_yd2 = ['oz/sy', 'oz/yd2']

densityarea_units = kg_m2 + kg_yd2 + psf + kg_dm2 + t_m2 + g_m2 + oz_yd2

def density2kgm2(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in kg/m2)
    """
    if unit in kg_m2:
        pass
    
    elif unit in kg_yd2:
        qty *= 1.09361**2
        
    elif unit in psf:
        qty *= 3.28084**2/2.20462
        
    elif unit in kg_dm2:
        qty *= 100
    
    elif unit in t_m2:
        qty *= 1_000
    
    elif unit in g_m2:
        qty /= 1_000
    
    elif unit in oz_yd2:
        qty *= 1.09361**2/35.274
    

        
    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the density2kgm2 function')
        
    return qty

################################### density2kgm3 #############################################################################
################################### density2kgm3 #############################################################################

kg_m3 = ['kg/m³', 'kg/m3', 'kg/m^3']
kg_yd3 = ['kg/yd3', 'kg/yd³']
lb_ft3 = ['lbs/ft3', 'lb/ft3', 'lb/ft³', 'pcf']
kg_dm3 = ['kg/dm3']
t_m3 = ['t/m3', 't/m³']
g_m3 = ['g/m3', 'g/m^3']
kg_l = ['kg/l', 'kg/liter', 'kg/litre']

density_units = kg_m3 + kg_yd3 + lb_ft3 + kg_dm3 + t_m3 + g_m3 + kg_l


def density2kgm3(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in m3)
    """
    if unit in kg_m3:
        pass
    
    elif unit in kg_yd3:
        qty *= 1/27/m3_ft3
        
    elif unit in lb_ft3:
        qty *= 1/m3_ft3/2.20462
        
    elif unit in kg_dm3:
        qty *= 1000
    
    elif unit in t_m3:
        qty*= 1000
    
    elif unit in g_m3:
        qty /= 1_000
        
    elif unit in kg_l:
        qty *= 1_000
        
    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the density2kgm3 function')
        
    return qty

############################## emission2kgco2e ##################################################################################
############################## emission2kgco2e ##################################################################################

lbco2e = ['lbco2e']
tco2e = ['tco2e']
kgco2e = ['kgco2e']
emission_units = lbco2e + tco2e + kgco2e


def emission2kgco2e(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in psi)
    """
    if unit in lbco2e:
        qty /= 2.20462
    
    elif unit in tco2e:
        qty *= 1_000
        
    elif unit in kgco2e:
        pass

    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the emission2kgco2e function')
        
    return qty


########################### emission2kgmwh #####################################################################################
########################### emission2kgmwh #####################################################################################

lbco2mwh = ['lbco2e/mwh']
tco2mwh = ['tco2e/mwh']
kgco2mwh = ['kgco2e/mwh']
energyemission_units = lbco2mwh + tco2mwh + kgco2mwh

def emission2kgmwh(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in psi)
    """
    if unit in lbco2mwh:
        qty /= 2.20462
    
    elif unit in tco2mwh:
        qty *= 1_000
        
    elif unit in kgco2mwh:
        pass
    
    
    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the emission2kgmwh function')
        
    return qty

############################# length2in ###################################################################################
############################# length2in ###################################################################################

m1 = ['m', 'meter', 'meters']
ft = ['ft' ,'feet', 'fts']
cm = ['cm', 'centimeter', 'cms']
mm = ['mm', 'milli', 'millimeter', 'millimeters', 'mms', 'mil']
inch = ['in', 'inches', 'inch', 'in.']
km = ['km', 'kilometer', 'kilometers', 'kms']
length_units = m1 + ft + cm + mm + inch + km


def length2in(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in psi)
    """
    if unit in m1:
        qty *= 39.3701
    
    elif unit in ft:
        qty *= 12
        
    elif unit in cm:
        qty /= 2.54

    elif unit in mm:
        qty /= 25.4
    
    elif unit in inch:
        pass
    
    elif unit in km:
        qty *= 1000*39.3701
    
    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the length2in function')
        
    return qty

############################## pressure2psi ##################################################################################
############################## pressure2psi ##################################################################################

psi = ['psi', 'lb/in2', 'lbs/in2']
ksi = ['ksi', 'k/in2']
mpa = ['mpa']
nmm2 = ['n/mm2', 'n/mm*2']
pressure_units = psi + ksi + mpa + nmm2


def pressure2psi(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in psi)
    """
    if unit in psi:
        pass
    
    elif unit in ksi:
        qty *= 1_000
        
    elif unit in mpa:
        qty *= psi_mpa
    
    elif unit in nmm2:
        qty *= 145.038

    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the pressure2psi function')
        
    return qty


################################ therm2rval ################################################################################
################################ therm2rval ################################################################################

uval = ['u_value', 'u_val', 'uvalue', 'uval']
rval = ['r_value', 'u_val', 'rvalue', 'rval']
rsival = ['rsi', 'rsi_val']
therm_units = uval + rval + rsival


def therm2rval(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in r-value)
    """
    if unit in uval:
        qty = 1/qty
    
    elif unit in rval:
        pass
    
    elif unit in rsival:
        qty *= 5.6785917092561045

    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the therm2rval function')
        
    return qty


################################ time2year ################################################################################
################################ time2year ################################################################################

decade = ['decades', 'decade', 'dec']
year = ['yr', 'year', 'years', 'yrs']
day = ['days', 'day']
hour = ['hrs', 'hours', 'hr', 'hour']
minute = ['min', 'mins', 'minutes', 'minute']
second = ['second', 'seconds', 'secs', 'sec']
time_units = decade + year + day + hour + minute + second


def time2year(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in psi)
    """
    if unit in decade:
        qty *= 10
    
    elif unit in year:
        pass
        
    elif unit in day:
        qty /= 365 
    
    elif unit in hour:
        qty /= 365*24

    elif unit in minute:
        qty /= 365*24*60

    elif unit in second:
        qty /= 365*24*60*60

    else:
        qty = None
        if prnt=='y':
            print(f'{unit} is not being accounted for in the time2year function')
        
    return qty







################################ vol2m3 ################################################################################
################################ vol2m3 ################################################################################

m3 = ['m3', 'm³', 'm^3']
ft3 = ['ft3', 'ft^3', 'cuft', 'ft³']
yd3 = ['yd3', 'yd³', 'cy', 'yd^3']
volume_units = ft3 + m3 + yd3


def vol2m3(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in m3)
    """
    if unit in m3:
        pass
    
    elif unit in ft3:
        qty *= m3_ft3
        
    elif unit in yd3:
        qty *= 27*m3_ft3
        
    else:
        qty = None
        if prnt == 'y':
            print(f'{unit} is not being accounted for in the vol2m3 function')
    
    return qty


################################## weight2kgs ##############################################################################
################################## weight2kgs ##############################################################################

kgs = ['kg', 'kgs']
gs = ['g', 'gs']
lbs = ['lb', 'lbs', 'pound', 'pounds']
tons = ['t', 'ton', 'short', 'shortton']
tonnes = ['metric', 'tonne', 'metricton']
weight_units = kgs + gs + lbs + tons + tonnes

def weight2kgs(qty, unit, prnt='y'):
    """
    INPUTS: (qty, unit, prnt='y')
    OUTPUTS: qty (in kg)
    """
    if unit in kgs:
        pass
        
    elif unit in gs:
        qty /= 1000
        
    elif unit in lbs:
        qty /= 2.20462
    
    elif unit in tons:
        qty *= 2000/2.20462
        
    elif unit in tonnes:
        qty *= 1000
        
    else:
        qty = None
        if prnt == 'y':
            print(f'{unit} is not being accounted for in the weight2kgs function')
        
    return qty

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


############################# str2valunit ###################################################################################
############################# str2valunit ###################################################################################

def str2valunit(string):
    """
    INPUTS: (string)
    OUTPUTS: value, unit
    """
    if type(string) != str:
        return np.nan, np.nan
    string = string.replace(' ','').lower()
    string = string.replace(',','.')
    tf_list = [any([ele.isdigit(), ele=='.', ele=='-', ele=='+']) for ele in list(string)]
    e_loc = string.find('e')
    if e_loc != -1 and e_loc != len(string)-1 and tf_list[e_loc+1] == True: #in case the value is in the format 1234e-2
        tf_list[e_loc] = True
    
    if True not in tf_list:
        return np.nan, np.nan
    elif False in tf_list:
        x = tf_list.index(False)
        value = float(string[:x])
        unit = string[x:]
        return value, unit
    else:
        return np.nan, np.nan
    
######################## consistent_units ########################################################################################
######################## consistent_units ########################################################################################

dict_unitconv = {
    'area': {'func':area2m2, 'units': area_units, 'output':'m2'},
    'density_area': {'func':density2kgm2, 'units': densityarea_units, 'output':'kgm-2'},
    'density_vol': {'func':density2kgm3, 'units': density_units, 'output':'kgm-3'},
    'emissions': {'func':emission2kgco2e, 'units': emission_units, 'output':'kgco2e'},
    'emissions_energy': {'func':emission2kgmwh, 'units': energyemission_units, 'output':'kgmwh-1'},
    'length': {'func':length2in, 'units': length_units, 'output':'in'},
    'pressure': {'func':pressure2psi, 'units': pressure_units, 'output':'psi'},
    'thermalresistance': {'func': therm2rval, 'units': therm_units, 'output': 'rval'},
    'time': {'func':time2year, 'units': time_units, 'output':'yr'},
    'vol': {'func':vol2m3, 'units': volume_units, 'output':'m3'},
    'weight': {'func':weight2kgs,'units': weight_units, 'output':'kg'},
}


#interpret each unit by type
dict_unittype = {}
for typ in dict_unitconv:
    for unit in dict_unitconv[typ]['units']:
        dict_unittype[unit] = typ


def consistent_units(input_list):
    """
    INPUTS: an input_list with strings, such as '6 MPa'
    OUTPUTS: list(converted_values), str(unit), list(unaccounted_units)
    """
    dft = pd.DataFrame(columns=['input', 'qty', 'unit'])
    unaccounted_units = []
    dft['input'] = input_list
    dft[['qty', 'unit']] = pd.DataFrame(dft['input'].astype(str).apply(str2valunit).tolist(), index=dft.index)
    dft['type'] = dft['unit'].map(dict_unittype)
    if dft['type'].isnull().all():
        return list(input_list), 'unknown', list(dft['unit'].unique())
    unittype = dft['type'].value_counts().index[0]
    dft['output'] = dft.apply(lambda x: dict_unitconv[unittype]['func'](x['qty'], x['unit'], prnt='n'), axis=1)
    unaccounted_units = list(dft[dft['type'].isnull()]['unit'].unique())
    
    return list(dft['output']), dict_unitconv[unittype]['output'], unaccounted_units