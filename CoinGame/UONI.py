from board import Board #line:1
from agent import Agent #line:2
import random #line:3
class UONI (Agent ):#line:5
    def __init__ (O000O00000000O00O ,player =1 ,level ='easy'):#line:6
        super ().__init__ (player )#line:7
        if level =='easy':#line:8
            O000O00000000O00O .p =0.6 #line:9
        elif level =='medium':#line:10
            O000O00000000O00O .p =0.3 #line:11
        elif level =='hard':#line:12
            O000O00000000O00O .p =0.0 #line:13
    def next_action (OO00O0O0O0O0O0OO0 ,OOOOO0O00O0OO00O0 ):#line:15
        if random .random ()<=OO00O0O0O0O0O0OO0 .p :#line:16
            O0OOO0000000O0OOO =OOOOO0O00O0OO00O0 .get_possible_actions ()#line:17
            return random .choice (O0OOO0000000O0OOO )#line:18
        O0O0OOOOOOO000OO0 ,_O00O00000OO0O0O00 =OO00O0O0O0O0O0OO0 .mei0mfu9430u53905v (OOOOO0O00O0OO00O0 ,OO00O0O0O0O0O0OO0 .player ,2 ,float ('-inf'),float ('inf'))#line:19
        return O0O0OOOOOOO000OO0 #line:20
    def heuristic_utility (OOO0OO0OO00O00000 ,O0O0OO00O0O0OO00O ):#line:22
        return OOO0OO0OO00O00000 .ansjfkashjrfieoawhfie923u492 (O0O0OO00O0O0OO00O )#line:23
    def ansjfkashjrfieoawhfie923u492 (O00O0OOO0O00OO0O0 ,O0OOO0OO00OOOOO0O ):#line:25
        O00000OO0O0O00000 =0 #line:26
        for O0O00OOOO0O0000OO in O0OOO0OO00OOOOO0O .grid :#line:27
            OOO0OOOOOOO00000O =0 #line:28
            for O0O0O0OO0O00OOO00 in O0O00OOOO0O0000OO :#line:29
                if O0O0O0OO0O00OOO00 ==0 :#line:30
                    if OOO0OOOOOOO00000O !=0 :#line:31
                        O00000OO0O0O00000 ^=OOO0OOOOOOO00000O #line:32
                        OOO0OOOOOOO00000O =0 #line:33
                else :#line:34
                    OOO0OOOOOOO00000O +=1 #line:35
            if OOO0OOOOOOO00000O ==sum (O0O00OOOO0O0000OO ):#line:36
                O00000OO0O0O00000 ^=OOO0OOOOOOO00000O #line:37
        return -O00000OO0O0O00000 #line:38
    def mivc90v4urt985v891v5vf5c12f (O0O000O00000OO0OO ,OO0O00OO0000O0O0O ):#line:40
        O00000000O00OO00O =0 #line:41
        for OO000OOO00OO00O0O in OO0O00OO0000O0O0O .grid :#line:42
            O000OOOO0OOOO00O0 =0 #line:43
            for O00000OOOO00OOO0O in OO000OOO00OO00O0O :#line:44
                if O00000OOOO00OOO0O ==0 :#line:45
                    if O000OOOO0OOOO00O0 !=0 :#line:46
                        O00000000O00OO00O =max (O000OOOO0OOOO00O0 ,O00000000O00OO00O )#line:47
                        O000OOOO0OOOO00O0 =0 #line:48
                else :#line:49
                    O000OOOO0OOOO00O0 +=1 #line:50
            if O000OOOO0OOOO00O0 ==sum (OO000OOO00OO00O0O ):#line:51
                O00000000O00OO00O =max (O000OOOO0OOOO00O0 ,O00000000O00OO00O )#line:52
        return -O00000000O00OO00O #line:53
    def a473nmv8975721589 (OO0O0O0O0O0OOO0OO ,OOO00O0OOOOOO0000 ):#line:55
        O0O00000000O00OO0 =0 #line:56
        for OO0O0000000O000OO in OOO00O0OOOOOO0000 .grid :#line:57
            O00OOO0OOO0OO00O0 =0 #line:58
            for OOO000O0OOOO000O0 in OO0O0000000O000OO :#line:59
                if OOO000O0OOOO000O0 ==0 :#line:60
                    if O00OOO0OOO0OO00O0 !=0 :#line:61
                        O0O00000000O00OO0 +=1 #line:62
                        O00OOO0OOO0OO00O0 =0 #line:63
                else :#line:64
                    O00OOO0OOO0OO00O0 +=1 #line:65
            if O00OOO0OOO0OO00O0 ==sum (OO0O0000000O000OO ):#line:66
                O0O00000000O00OO0 =1 #line:67
        return -O0O00000000O00OO0 #line:68
    def mcu90589341nv89 (O00OOOOOOOO0OO0O0 ,OOOO00OO000O0O0OO ,OO0OO00000OO00O0O ):#line:70
        OOOOO0OOO0O00OO0O =len (OO0OO00000OO00O0O )#line:71
        return -OOOOO0OOO0O00OO0O #line:72
    def mivc90v4urt985v891v5vf5c12f (O000O0O0O000000O0 ,O0O0OOO0O0O0O0OO0 ):#line:74
        OO0O000000OO0OO00 =0 #line:75
        for OOOOO000O0OO00000 in O0O0OOO0O0O0O0OO0 .grid :#line:76
            O00000O0O00000O0O =len (OOOOO000O0OO00000 )#line:77
            if O00000O0O00000O0O >OO0O000000OO0OO00 :#line:78
                OO0O000000OO0OO00 =O00000O0O00000O0O #line:79
        return -OO0O000000OO0OO00 #line:80
    def a473nmv8975721589 (O00O0O0OOOO00OO00 ,OOOOOOO00O0000OO0 ):#line:82
        O000000O00OOOO000 =0 #line:83
        for OOOOO000000OO00O0 in OOOOOOO00O0000OO0 .grid :#line:84
            if len (OOOOO000000OO00O0 )>0 :#line:85
                O000000O00OOOO000 +=1 #line:86
        return -O000000O00OOOO000 #line:87
    def ads2vn5879v5m3 (O0OOO00OOOO0O000O ,O00OO000OO0OOOO0O ):#line:89
        return -O00OO000OO0OOOO0O .grid .sum ()#line:90
    def mei0mfu9430u53905v (O0O0OOOO00OO0OOOO ,OOO000OOO0OOOO000 ,OOOOO0OO000000OOO ,OO0OO0OOO00O0O0O0 ,O000OOOOOO0OO0OO0 ,OOOO000O0O0O0OOOO ):#line:92
        O0000O0OOO00OOOOO ,O0O00O0000OO0O0OO =OOO000OOO0OOOO000 .is_end (OOOOO0OO000000OOO )#line:94
        if O0000O0OOO00OOOOO :#line:95
            if O0O00O0000OO0O0OO ==O0O0OOOO00OO0OOOO .player :#line:96
                return None ,1 #line:97
            elif O0O00O0000OO0O0OO ==2 :#line:98
                return None ,-1 #line:99
            else :#line:100
                return None ,0 #line:101
        if OO0OO0OOO00O0O0O0 ==0 :#line:103
            return None ,O0O0OOOO00OO0OOOO .heuristic_utility (OOO000OOO0OOOO000 )#line:104
        O0O00O0OO00O0OOOO =OOO000OOO0OOOO000 .get_possible_actions ()#line:107
        OOO0OO00OO00000OO =[]#line:108
        for O0O0OO0O0000OOO00 in O0O00O0OO00O0OOOO :#line:109
            OO0OO00OO0O0O000O =OOO000OOO0OOOO000 .clone ()#line:110
            O0OOO0000OO0O00O0 =OO0OO00OO0O0O000O .play (O0O0OO0O0000OOO00 )#line:111
            if not O0OOO0000OO0O00O0 :#line:112
                raise Exception ("O0O00O0OO00O0OOOO")#line:113
            OOO0OO00OO00000OO .append ((O0O0OO0O0000OOO00 ,OO0OO00OO0O0O000O ))#line:114
        OOOO0000OOO0OOOO0 =None #line:116
        if OOOOO0OO000000OOO !=O0O0OOOO00OO0OOOO .player :#line:118
            OO00O0OO0O00OO000 =float ('inf')#line:119
            for O0O0OO0O0000OOO00 ,OO0OOOO0OOO0OOOOO in OOO0OO00OO00000OO :#line:120
                _OO0OOO0O00000000O ,OOO0OO0000OOO0O0O =O0O0OOOO00OO0OOOO .mei0mfu9430u53905v (OO0OOOO0OOO0OOOOO ,(OOOOO0OO000000OOO %2 )+1 ,OO0OO0OOO00O0O0O0 -1 ,O000OOOOOO0OO0OO0 ,OOOO000O0O0O0OOOO )#line:121
                if OO00O0OO0O00OO000 >OOO0OO0000OOO0O0O :#line:122
                    OO00O0OO0O00OO000 =OOO0OO0000OOO0O0O #line:123
                    OOOO0000OOO0OOOO0 =O0O0OO0O0000OOO00 #line:124
                O000OOOOOO0OO0OO0 =max (O000OOOOOO0OO0OO0 ,OO00O0OO0O00OO000 )#line:125
                if O000OOOOOO0OO0OO0 >=OOOO000O0O0O0OOOO :#line:126
                    break #line:127
            return OOOO0000OOO0OOOO0 ,OO00O0OO0O00OO000 #line:128
        else :#line:129
            OO000OOOOOOOO00OO =float ('-inf')#line:130
            for O0O0OO0O0000OOO00 ,OO0OOOO0OOO0OOOOO in OOO0OO00OO00000OO :#line:131
                _OO0OOO0O00000000O ,OOO0OO0000OOO0O0O =O0O0OOOO00OO0OOOO .mei0mfu9430u53905v (OO0OOOO0OOO0OOOOO ,(OOOOO0OO000000OOO %2 )+1 ,OO0OO0OOO00O0O0O0 -1 ,O000OOOOOO0OO0OO0 ,OOOO000O0O0O0OOOO )#line:132
                if OOO0OO0000OOO0O0O >OO000OOOOOOOO00OO :#line:133
                    OO000OOOOOOOO00OO =OOO0OO0000OOO0O0O #line:134
                    OOOO0000OOO0OOOO0 =O0O0OO0O0000OOO00 #line:135
                OOOO000O0O0O0OOOO =min (OOOO000O0O0O0OOOO ,OO000OOOOOOOO00OO )#line:136
                if O000OOOOOO0OO0OO0 >=OOOO000O0O0O0OOOO :#line:137
                    break #line:138
            return OOOO0000OOO0OOOO0 ,OO000OOOOOOOO00OO 