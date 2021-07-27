import pandas as pd
import numpy as np
import pickle
import sys
import os

def predict_price(Year,Mileage,State,Make,Model):

    State = State.replace(State,' '+State)
    
    df_predict = pd.DataFrame({'Year':[Year], 'Mileage':[Mileage], 'State':[State], 'Make':[Make], 'Model':[Model]})
    
    state_list = [' TX', ' CA', ' FL', ' GA', ' NC', ' IL', ' VA', ' PA', ' NY', ' NJ',
       ' OH', ' CO', ' WA', ' AZ', ' TN', ' MA', ' MD', ' MO', ' IN', ' KY']

    model_list = ['Silverado', 'Grand', 'Sierra', 'Accord', 'F-1504WD', 'Wrangler',
       'Civic', '3', 'Jetta', 'Santa', 'FusionSE', 'EquinoxFWD',
       'CamrySE', 'Super', 'CorollaLE', 'MalibuLT', 'F-150XLT', 'Town',
       'Ram', 'CamaroCoupe', 'Cooper', 'SorentoLX', 'ExplorerXLT',
       'Rover', 'EscapeSE', 'OptimaLX', '5', 'Tundra', 'RX', 'EquinoxAWD',
       'F-1502WD', 'CamryLE', 'CR-VEX-L', 'Passat4dr', 'TerrainFWD',
       'CruzeSedan', 'Elantra4dr', 'Outback2.5i', 'CorollaS',
       '200Limited', 'EscapeFWD', 'EdgeSEL', 'AcadiaFWD', 'ES',
       'E-ClassE350', 'FocusSE', 'Tahoe4WD', 'Sonata4dr', 'AcadiaAWD',
       'CTS', 'CR-VLX', 'IS', 'OdysseyEX-L', 'C-ClassC300', 'MuranoAWD',
       'Camry4dr', 'Yukon', 'TraverseFWD', 'CR-VEX', 'Tacoma4WD',
       'Escape4WD', 'CivicLX', 'PatriotSport', 'Malibu1LT', 'G37',
       'Altima4dr', 'Tacoma2WD', 'RXRX', 'Tahoe2WD', 'ExplorerLimited',
       'Transit', 'MalibuLS', 'TerrainAWD', 'EnclaveLeather', 'PilotEX-L',
       'F-150Lariat', 'Impreza', 'Suburban4WD', 'SRXLuxury', 'Prius',
       '25004WD', 'Yukon4WD', 'AccordEX-L', 'Explorer4WD', 'CompassSport',
       'Soul+', 'OptimaEX', 'TraverseAWD', 'F-150SuperCrew', 'Legacy2.5i',
       'SedonaLX', 'ForteLX', 'Mustang2dr', 'CamryXLE', 'CorollaL',
       'MustangGT', 'CherokeeLimited', 'LaCrosseFWD', 'SonataSE',
       'PatriotLatitude', 'ChallengerR/T', 'M-ClassML350', 'ChargerSXT',
       'Fusion4dr', 'JourneySXT', 'Express', 'ESES', 'JourneyFWD',
       'Corvette2dr', 'F-150XL', 'Golf', 'Wrangler4WD', '15004WD',
       'ColoradoCrew', 'CompassLatitude', 'Corolla4dr', 'RAV4XLE',
       'ImpalaLT', 'AccordLX', 'SportageLX', 'ISIS', 'SonataLimited',
       'TacomaBase', 'Suburban2WD', 'WranglerSport', 'C-ClassC',
       'PriusTwo', 'Elantra', '300300C', 'Malibu', 'Yukon2WD',
       'C-Class4dr', 'Escalade', 'TahoeLT', 'X3xDrive28i', 'CR-V4WD',
       'Charger4dr', 'Outlander', 'Malibu4dr', 'TucsonFWD', 'MDXAWD',
       'SonicSedan', 'EdgeLimited', 'GS', 'SiennaXLE', 'Camaro2dr',
       'RAV4LE', 'Expedition', 'Pilot4WD', 'OdysseyTouring', 'Murano2WD',
       'FusionHybrid', '7', '4Runner4WD', 'CamaroConvertible',
       'Impala4dr', 'ElantraLimited', 'TundraSR5', 'Challenger2dr',
       '4RunnerSR5', 'ChargerSE', 'Prius5dr', 'AvalonXLE',
       'HighlanderFWD', 'DurangoAWD', 'GX', 'Maxima4dr', 'Sienna5dr',
       'RAV4Limited', 'X5xDrive35i', 'CivicEX', 'ExplorerFWD',
       'Econoline', 'TaurusSEL', 'Pathfinder4WD', 'X5AWD', 'PriusThree',
       'Camry', 'RAV44WD']

    make_list = ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'Jeep', 'GMC', 'Kia', 'Dodge',
       'Hyundai', 'Lexus', 'BMW']

    df_predict['State_new']=np.where(df_predict['State'].isin(state_list),df_predict['State'],"OTRO")
    df_predict['Model_new']=np.where(df_predict['Model'].isin(model_list),df_predict['Model'],"OTRO")
    df_predict['Make_new']=np.where(df_predict['Make'].isin(make_list),df_predict['Make'],"OTRO")
    
    state_list.append('OTRO')
    model_list.append('OTRO')
    make_list.append('OTRO')

    df_predict['State_new'] = pd.Categorical(
                       df_predict.State_new,
                       state_list
                       )

    df_predict['Model_new'] = pd.Categorical(
                       df_predict.Model_new,
                       model_list
                       )

    df_predict['Make_new'] = pd.Categorical(
                       df_predict.Make_new,
                       make_list
                       )

    data = df_predict[['Year','Mileage','State_new','Model_new','Make_new']]
    data = pd.get_dummies(data)
    
    data = data[['Year', 'Mileage', 'State_new_ AZ', 'State_new_ CA',
       'State_new_ CO', 'State_new_ FL', 'State_new_ GA', 'State_new_ IL',
       'State_new_ IN', 'State_new_ KY', 'State_new_ MA', 'State_new_ MD',
       'State_new_ MO', 'State_new_ NC', 'State_new_ NJ', 'State_new_ NY',
       'State_new_ OH', 'State_new_ PA', 'State_new_ TN', 'State_new_ TX',
       'State_new_ VA', 'State_new_ WA', 'State_new_OTRO',
       'Model_new_15004WD', 'Model_new_200Limited', 'Model_new_25004WD',
       'Model_new_3', 'Model_new_300300C', 'Model_new_4Runner4WD',
       'Model_new_4RunnerSR5', 'Model_new_5', 'Model_new_7',
       'Model_new_AcadiaAWD', 'Model_new_AcadiaFWD', 'Model_new_Accord',
       'Model_new_AccordEX-L', 'Model_new_AccordLX',
       'Model_new_Altima4dr', 'Model_new_AvalonXLE',
       'Model_new_C-Class4dr', 'Model_new_C-ClassC',
       'Model_new_C-ClassC300', 'Model_new_CR-V4WD', 'Model_new_CR-VEX',
       'Model_new_CR-VEX-L', 'Model_new_CR-VLX', 'Model_new_CTS',
       'Model_new_Camaro2dr', 'Model_new_CamaroConvertible',
       'Model_new_CamaroCoupe', 'Model_new_Camry', 'Model_new_Camry4dr',
       'Model_new_CamryLE', 'Model_new_CamrySE', 'Model_new_CamryXLE',
       'Model_new_Challenger2dr', 'Model_new_ChallengerR/T',
       'Model_new_Charger4dr', 'Model_new_ChargerSE',
       'Model_new_ChargerSXT', 'Model_new_CherokeeLimited',
       'Model_new_Civic', 'Model_new_CivicEX', 'Model_new_CivicLX',
       'Model_new_ColoradoCrew', 'Model_new_CompassLatitude',
       'Model_new_CompassSport', 'Model_new_Cooper',
       'Model_new_Corolla4dr', 'Model_new_CorollaL',
       'Model_new_CorollaLE', 'Model_new_CorollaS',
       'Model_new_Corvette2dr', 'Model_new_CruzeSedan',
       'Model_new_DurangoAWD', 'Model_new_E-ClassE350', 'Model_new_ES',
       'Model_new_ESES', 'Model_new_Econoline', 'Model_new_EdgeLimited',
       'Model_new_EdgeSEL', 'Model_new_Elantra', 'Model_new_Elantra4dr',
       'Model_new_ElantraLimited', 'Model_new_EnclaveLeather',
       'Model_new_EquinoxAWD', 'Model_new_EquinoxFWD',
       'Model_new_Escalade', 'Model_new_Escape4WD', 'Model_new_EscapeFWD',
       'Model_new_EscapeSE', 'Model_new_Expedition',
       'Model_new_Explorer4WD', 'Model_new_ExplorerFWD',
       'Model_new_ExplorerLimited', 'Model_new_ExplorerXLT',
       'Model_new_Express', 'Model_new_F-1502WD', 'Model_new_F-1504WD',
       'Model_new_F-150Lariat', 'Model_new_F-150SuperCrew',
       'Model_new_F-150XL', 'Model_new_F-150XLT', 'Model_new_FocusSE',
       'Model_new_ForteLX', 'Model_new_Fusion4dr',
       'Model_new_FusionHybrid', 'Model_new_FusionSE', 'Model_new_G37',
       'Model_new_GS', 'Model_new_GX', 'Model_new_Golf',
       'Model_new_Grand', 'Model_new_HighlanderFWD', 'Model_new_IS',
       'Model_new_ISIS', 'Model_new_Impala4dr', 'Model_new_ImpalaLT',
       'Model_new_Impreza', 'Model_new_Jetta', 'Model_new_JourneyFWD',
       'Model_new_JourneySXT', 'Model_new_LaCrosseFWD',
       'Model_new_Legacy2.5i', 'Model_new_M-ClassML350',
       'Model_new_MDXAWD', 'Model_new_Malibu', 'Model_new_Malibu1LT',
       'Model_new_Malibu4dr', 'Model_new_MalibuLS', 'Model_new_MalibuLT',
       'Model_new_Maxima4dr', 'Model_new_Murano2WD',
       'Model_new_MuranoAWD', 'Model_new_Mustang2dr',
       'Model_new_MustangGT', 'Model_new_OTRO', 'Model_new_OdysseyEX-L',
       'Model_new_OdysseyTouring', 'Model_new_OptimaEX',
       'Model_new_OptimaLX', 'Model_new_Outback2.5i',
       'Model_new_Outlander', 'Model_new_Passat4dr',
       'Model_new_Pathfinder4WD', 'Model_new_PatriotLatitude',
       'Model_new_PatriotSport', 'Model_new_Pilot4WD',
       'Model_new_PilotEX-L', 'Model_new_Prius', 'Model_new_Prius5dr',
       'Model_new_PriusThree', 'Model_new_PriusTwo', 'Model_new_RAV44WD',
       'Model_new_RAV4LE', 'Model_new_RAV4Limited', 'Model_new_RAV4XLE',
       'Model_new_RX', 'Model_new_RXRX', 'Model_new_Ram',
       'Model_new_Rover', 'Model_new_SRXLuxury', 'Model_new_Santa',
       'Model_new_SedonaLX', 'Model_new_Sienna5dr', 'Model_new_SiennaXLE',
       'Model_new_Sierra', 'Model_new_Silverado', 'Model_new_Sonata4dr',
       'Model_new_SonataLimited', 'Model_new_SonataSE',
       'Model_new_SonicSedan', 'Model_new_SorentoLX', 'Model_new_Soul+',
       'Model_new_SportageLX', 'Model_new_Suburban2WD',
       'Model_new_Suburban4WD', 'Model_new_Super', 'Model_new_Tacoma2WD',
       'Model_new_Tacoma4WD', 'Model_new_TacomaBase',
       'Model_new_Tahoe2WD', 'Model_new_Tahoe4WD', 'Model_new_TahoeLT',
       'Model_new_TaurusSEL', 'Model_new_TerrainAWD',
       'Model_new_TerrainFWD', 'Model_new_Town', 'Model_new_Transit',
       'Model_new_TraverseAWD', 'Model_new_TraverseFWD',
       'Model_new_TucsonFWD', 'Model_new_Tundra', 'Model_new_TundraSR5',
       'Model_new_Wrangler', 'Model_new_Wrangler4WD',
       'Model_new_WranglerSport', 'Model_new_X3xDrive28i',
       'Model_new_X5AWD', 'Model_new_X5xDrive35i', 'Model_new_Yukon',
       'Model_new_Yukon2WD', 'Model_new_Yukon4WD', 'Make_new_BMW',
       'Make_new_Chevrolet', 'Make_new_Dodge', 'Make_new_Ford',
       'Make_new_GMC', 'Make_new_Honda', 'Make_new_Hyundai',
       'Make_new_Jeep', 'Make_new_Kia', 'Make_new_Lexus', 'Make_new_OTRO',
       'Make_new_Toyota']]
    
    price = xgboost.predict(data)[0]
    
    return price


#if __name__ == "__main__":
    
#    if len(sys.argv) == 1:
#        print('Please add features:')
#       
#   else:
#       # Year,Mileage,State,Make,Model = sys.argv[1,2,3,4,5]

#       price = predict_price(Year,Mileage,State,Make,Model)
#       
#       print(Year,Mileage,State,Make,Model)
#       print('Price of car: ', price)
