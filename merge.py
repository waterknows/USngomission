import pandas as pd
df1=pd.read_table('bmf.txt', header=None,names=['EIN','NAME','SUBSECTION','NTEE'],index_col='EIN')
df2=pd.read_table('crosswalk.txt',sep='|')
df2.columns=['XML','EIN']
df3=pd.read_table('miss.txt',header=None,names=['XML','MISSION','CONTRIBUTION','SERVICE','BEGIN_YEAR','END_YEAR'],na_values=.0000)
df3['YEAR']=df3['BEGIN_YEAR'].str[:4].astype(int)
print('files imported')
df32=pd.merge(df3,df2,how='left',on='XML')
fin=df32[['EIN','CONTRIBUTION','SERVICE','YEAR']].dropna(axis=0,how='any').groupby('EIN').sum()
fin.columns=['CONTRIBUTION','SERVICE','TOTYEAR']
print('merging financial info')
df4=df1.join(fin,how='left')
mis=df32[['EIN','MISSION','YEAR']].sort_values(by='YEAR').drop_duplicates(subset='EIN',keep='last').set_index('EIN')
print('merging mission info')
df5=df4.join(mis,how='left')
print('merging finished,dropping missing',df5.columns,len(df5))
df6=df5.dropna(axis=0,how='any')
print('saving to ngo.csv...total rows',len(df6))
df6.to_csv('ngo.txt',sep='\t')
