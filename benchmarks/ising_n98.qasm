OPENQASM 2.0;
include "qelib1.inc";
qreg q[98];
u3(pi/2,-0.10547852679489633,-pi) q[0];
u3(0.21095697320510337,-pi/2,pi/2) q[1];
cz q[0],q[1];
u3(0.1054784900000001,pi/2,-pi/2) q[1];
cz q[0],q[1];
u3(pi/2,-0.3166508000000001,-pi) q[1];
u3(pi/2,-0.3211291267948968,-pi) q[2];
u3(0.6422581732051035,-pi/2,pi/2) q[3];
cz q[2],q[3];
u3(0.3211290900000001,pi/2,-pi/2) q[3];
cz q[2],q[3];
u3(pi/2,0,-2.5082910535897933) q[2];
cz q[1],q[2];
u3(0.3166508000000002,pi/2,-pi/2) q[2];
cz q[1],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,1.6679692,-pi) q[3];
u3(pi/2,1.2551977732051034,-pi) q[4];
u3(2.5103955967948965,pi/2,-pi/2) q[5];
cz q[4],q[5];
u3(1.2551977999999997,-pi/2,pi/2) q[5];
cz q[4],q[5];
u3(pi/2,0,-0.19434575358979345) q[4];
cz q[3],q[4];
u3(1.6679691999999997,-pi/2,pi/2) q[4];
cz q[3],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0.94889695,-pi) q[5];
u3(pi/2,0.8327475732051033,-pi) q[6];
u3(1.6654950997948967,pi/2,-pi/2) q[7];
cz q[6],q[7];
u3(0.8327475499999999,-pi/2,pi/2) q[7];
cz q[6],q[7];
u3(pi/2,0,1.2437987535897932) q[6];
cz q[5],q[6];
u3(0.94889695,-pi/2,pi/2) q[6];
cz q[5],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,1.3389506000000004,-pi) q[7];
u3(pi/2,-0.5168785267948968,-pi) q[8];
u3(1.0337570732051033,-pi/2,pi/2) q[9];
cz q[8],q[9];
u3(0.5168785400000001,pi/2,-pi/2) q[9];
cz q[8],q[9];
u3(pi/2,0,0.46369145358979313) q[8];
cz q[7],q[8];
u3(1.3389505999999998,-pi/2,pi/2) q[8];
cz q[7],q[8];
u3(pi/2,0,pi) q[8];
u3(pi/2,0.42250814000000014,-pi) q[9];
u3(pi/2,-1.6689560997948962,-pi) q[10];
u3(2.9452731267948966,pi/2,-pi/2) q[11];
cz q[10],q[11];
u3(1.6689561000000002,pi/2,-pi/2) q[11];
cz q[10],q[11];
u3(pi/2,0,2.2965763735897937) q[10];
cz q[9],q[10];
u3(0.4225081399999998,-pi/2,pi/2) q[10];
cz q[9],q[10];
u3(pi/2,0,pi) q[10];
u3(pi/2,-0.1670009600000002,-pi) q[11];
u3(pi/2,-0.8778349567948966,-pi) q[12];
u3(1.7556698803846897,-pi/2,pi/2) q[13];
cz q[12],q[13];
u3(0.8778349599999999,pi/2,-pi/2) q[13];
cz q[12],q[13];
u3(pi/2,0,-2.807590733589793) q[12];
cz q[11],q[12];
u3(0.1670009600000001,pi/2,-pi/2) q[12];
cz q[11],q[12];
u3(pi/2,0,pi) q[12];
u3(pi/2,1.3028807000000002,-pi) q[13];
u3(pi/2,1.7917124803846898,-pi) q[14];
u3(2.6997602803846896,-pi/2,pi/2) q[15];
cz q[14],q[15];
u3(1.7917124999999998,-pi/2,pi/2) q[15];
cz q[14],q[15];
u3(pi/2,0,0.5358312535897931) q[14];
cz q[13],q[14];
u3(1.3028806999999998,-pi/2,pi/2) q[14];
cz q[13],q[14];
u3(pi/2,0,pi) q[14];
u3(pi/2,-1.8720307,-pi) q[15];
u3(pi/2,0.853188073205104,-pi) q[16];
u3(1.7063762167948966,pi/2,-pi/2) q[17];
cz q[16],q[17];
u3(0.8531881099999999,-pi/2,pi/2) q[17];
cz q[16],q[17];
u3(pi/2,0,0.6024687535897932) q[16];
cz q[15],q[16];
u3(1.8720307,pi/2,-pi/2) q[16];
cz q[15],q[16];
u3(pi/2,0,pi) q[16];
u3(pi/2,-1.1861875,-pi) q[17];
u3(pi/2,1.1007047732051038,-pi) q[18];
u3(2.2014095967948966,pi/2,-pi/2) q[19];
cz q[18],q[19];
u3(1.1007048,-pi/2,pi/2) q[19];
cz q[18],q[19];
u3(pi/2,0,-0.7692176535897932) q[18];
cz q[17],q[18];
u3(1.1861875000000002,pi/2,-pi/2) q[18];
cz q[17],q[18];
u3(pi/2,0,pi) q[18];
u3(pi/2,-0.11663793,-pi) q[19];
u3(pi/2,1.237413873205103,-pi) q[20];
u3(2.4748277967948966,pi/2,-pi/2) q[21];
cz q[20],q[21];
u3(1.2374138999999997,-pi/2,pi/2) q[21];
cz q[20],q[21];
u3(pi/2,0,-2.9083167935897936) q[20];
cz q[19],q[20];
u3(0.11663793000000013,pi/2,-pi/2) q[20];
cz q[19],q[20];
u3(pi/2,0,pi) q[20];
u3(pi/2,1.7434165000000004,-pi) q[21];
u3(pi/2,0.8690068732051035,-pi) q[22];
u3(1.7380138167948966,pi/2,-pi/2) q[23];
cz q[22],q[23];
u3(0.8690069099999999,-pi/2,pi/2) q[23];
cz q[22],q[23];
u3(pi/2,0,-0.34524035358979344) q[22];
cz q[21],q[22];
u3(1.7434164999999997,-pi/2,pi/2) q[22];
cz q[21],q[22];
u3(pi/2,0,pi) q[22];
u3(pi/2,1.9586679,-pi) q[23];
u3(pi/2,-0.07554432679489631,-pi) q[24];
u3(0.15108857320510335,-pi/2,pi/2) q[25];
cz q[24],q[25];
u3(0.07554429500000011,pi/2,-pi/2) q[25];
cz q[24],q[25];
u3(pi/2,0,-0.7757431535897936) q[24];
cz q[23],q[24];
u3(1.9586679,-pi/2,pi/2) q[24];
cz q[23],q[24];
u3(pi/2,0,pi) q[24];
u3(pi/2,-0.9186364299999998,-pi) q[25];
u3(pi/2,1.96669648038469,-pi) q[26];
u3(2.3497922803846896,-pi/2,pi/2) q[27];
cz q[26],q[27];
u3(1.9666964999999998,-pi/2,pi/2) q[27];
cz q[26],q[27];
u3(pi/2,0,-1.3043197535897932) q[26];
cz q[25],q[26];
u3(0.91863643,pi/2,-pi/2) q[26];
cz q[25],q[26];
u3(pi/2,0,pi) q[26];
u3(pi/2,-1.378703,-pi) q[27];
u3(pi/2,-0.8932566967948965,-pi) q[28];
u3(1.78651338038469,-pi/2,pi/2) q[29];
cz q[28],q[29];
u3(0.8932567,pi/2,-pi/2) q[29];
cz q[28],q[29];
u3(pi/2,0,-0.38418665358979354) q[28];
cz q[27],q[28];
u3(1.378703,pi/2,-pi/2) q[28];
cz q[27],q[28];
u3(pi/2,0,pi) q[28];
u3(pi/2,1.2285706000000003,-pi) q[29];
u3(pi/2,1.0753496732051033,-pi) q[30];
u3(2.150699396794897,pi/2,-pi/2) q[31];
cz q[30],q[31];
u3(1.0753497,-pi/2,pi/2) q[31];
cz q[30],q[31];
u3(pi/2,0,0.684451453589793) q[30];
cz q[29],q[30];
u3(1.2285705999999998,-pi/2,pi/2) q[30];
cz q[29],q[30];
u3(pi/2,0,pi) q[30];
u3(pi/2,0.2867082700000001,-pi) q[31];
u3(pi/2,-0.23544852679489647,-pi) q[32];
u3(0.47089697320510326,-pi/2,pi/2) q[33];
cz q[32],q[33];
u3(0.23544851000000006,pi/2,-pi/2) q[33];
cz q[32],q[33];
u3(pi/2,0,2.568176113589793) q[32];
cz q[31],q[32];
u3(0.28670826999999993,-pi/2,pi/2) q[32];
cz q[31],q[32];
u3(pi/2,0,pi) q[32];
u3(pi/2,0.19762375,-pi) q[33];
u3(pi/2,-0.1276773267948963,-pi) q[34];
u3(0.25535457320510335,-pi/2,pi/2) q[35];
cz q[34],q[35];
u3(0.12767731000000013,pi/2,-pi/2) q[35];
cz q[34],q[35];
u3(pi/2,0,2.746345153589793) q[34];
cz q[33],q[34];
u3(0.19762374999999988,-pi/2,pi/2) q[34];
cz q[33],q[34];
u3(pi/2,0,pi) q[34];
u3(pi/2,-0.20921184000000004,-pi) q[35];
u3(pi/2,-0.30930672679489657,-pi) q[36];
u3(0.6186134732051033,-pi/2,pi/2) q[37];
cz q[36],q[37];
u3(0.30930673000000003,pi/2,-pi/2) q[37];
cz q[36],q[37];
u3(pi/2,0,-2.7231689735897935) q[36];
cz q[35],q[36];
u3(0.20921184000000007,pi/2,-pi/2) q[36];
cz q[35],q[36];
u3(pi/2,0,pi) q[36];
u3(pi/2,0.01850870899999979,-pi) q[37];
u3(pi/2,0.7085020732051035,-pi) q[38];
u3(1.4170042367948967,pi/2,-pi/2) q[39];
cz q[38],q[39];
u3(0.7085021199999999,-pi/2,pi/2) q[39];
cz q[38],q[39];
u3(pi/2,0,3.1045752355897935) q[38];
cz q[37],q[38];
u3(0.01850870899999988,-pi/2,pi/2) q[38];
cz q[37],q[38];
u3(pi/2,0,pi) q[38];
u3(pi/2,-0.35743569,-pi) q[39];
u3(pi/2,0.38802257320510325,-pi) q[40];
u3(0.7760451167948966,pi/2,-pi/2) q[41];
cz q[40],q[41];
u3(0.3880225599999999,-pi/2,pi/2) q[41];
cz q[40],q[41];
u3(pi/2,0,-2.4267212735897936) q[40];
cz q[39],q[40];
u3(0.3574356900000001,pi/2,-pi/2) q[40];
cz q[39],q[40];
u3(pi/2,0,pi) q[40];
u3(pi/2,0.7844070200000002,-pi) q[41];
u3(pi/2,1.466366373205103,-pi) q[42];
u3(2.9327328267948967,pi/2,-pi/2) q[43];
cz q[42],q[43];
u3(1.4663663999999998,-pi/2,pi/2) q[43];
cz q[42],q[43];
u3(pi/2,0,1.5727786535897934) q[42];
cz q[41],q[42];
u3(0.7844070199999998,-pi/2,pi/2) q[42];
cz q[41],q[42];
u3(pi/2,0,pi) q[42];
u3(pi/2,1.9154827999999995,-pi) q[43];
u3(pi/2,-0.4883582267948965,-pi) q[44];
u3(0.9767163732051033,-pi/2,pi/2) q[45];
cz q[44],q[45];
u3(0.4883582100000001,pi/2,-pi/2) q[45];
cz q[44],q[45];
u3(pi/2,0,-0.689372953589793) q[44];
cz q[43],q[44];
u3(1.9154828,-pi/2,pi/2) q[44];
cz q[43],q[44];
u3(pi/2,0,pi) q[44];
u3(pi/2,1.0147279000000005,-pi) q[45];
u3(pi/2,-0.9882790567948967,-pi) q[46];
u3(1.9765580803846898,-pi/2,pi/2) q[47];
cz q[46],q[47];
u3(0.98827906,pi/2,-pi/2) q[47];
cz q[46],q[47];
u3(pi/2,0,1.112136853589793) q[46];
cz q[45],q[46];
u3(1.0147278999999998,-pi/2,pi/2) q[46];
cz q[45],q[46];
u3(pi/2,0,pi) q[46];
u3(pi/2,0.3730747700000001,-pi) q[47];
u3(pi/2,-0.34934642679489647,-pi) q[48];
u3(0.6986928732051035,-pi/2,pi/2) q[49];
cz q[48],q[49];
u3(0.34934642000000005,pi/2,-pi/2) q[49];
cz q[48],q[49];
u3(pi/2,0,2.395443113589793) q[48];
cz q[47],q[48];
u3(0.37307476999999983,-pi/2,pi/2) q[48];
cz q[47],q[48];
u3(pi/2,0,pi) q[48];
u3(pi/2,1.8344784999999995,-pi) q[49];
u3(pi/2,1.0319206732051036,-pi) q[50];
u3(2.0638413967948965,pi/2,-pi/2) q[51];
cz q[50],q[51];
u3(1.0319207,-pi/2,pi/2) q[51];
cz q[50],q[51];
u3(pi/2,0,-0.5273643535897934) q[50];
cz q[49],q[50];
u3(1.8344784999999997,-pi/2,pi/2) q[50];
cz q[49],q[50];
u3(pi/2,0,pi) q[50];
u3(pi/2,-1.9106685,-pi) q[51];
u3(pi/2,1.9122995803846896,-pi) q[52];
u3(2.4585860803846895,-pi/2,pi/2) q[53];
cz q[52],q[53];
u3(1.9122995999999999,-pi/2,pi/2) q[53];
cz q[52],q[53];
u3(pi/2,0,0.6797443535897929) q[52];
cz q[51],q[52];
u3(1.9106685000000003,pi/2,-pi/2) q[52];
cz q[51],q[52];
u3(pi/2,0,pi) q[52];
u3(pi/2,1.3210816000000003,-pi) q[53];
u3(pi/2,1.9124290803846895,-pi) q[54];
u3(2.4583270803846897,-pi/2,pi/2) q[55];
cz q[54],q[55];
u3(1.9124290999999998,-pi/2,pi/2) q[55];
cz q[54],q[55];
u3(pi/2,0,0.49942945358979296) q[54];
cz q[53],q[54];
u3(1.3210815999999999,-pi/2,pi/2) q[54];
cz q[53],q[54];
u3(pi/2,0,pi) q[54];
u3(pi/2,-0.7303085999999999,-pi) q[55];
u3(pi/2,-1.9350389967948967,-pi) q[56];
u3(2.4131073067948967,pi/2,-pi/2) q[57];
cz q[56],q[57];
u3(1.9350390000000002,pi/2,-pi/2) q[57];
cz q[56],q[57];
u3(pi/2,0,-1.6809754535897934) q[56];
cz q[55],q[56];
u3(0.7303086000000001,pi/2,-pi/2) q[56];
cz q[55],q[56];
u3(pi/2,0,pi) q[56];
u3(pi/2,0.3334761099999999,-pi) q[57];
u3(pi/2,1.328937073205104,-pi) q[58];
u3(2.6578742267948967,pi/2,-pi/2) q[59];
cz q[58],q[59];
u3(1.3289370999999999,-pi/2,pi/2) q[59];
cz q[58],q[59];
u3(pi/2,0,2.4746404335897934) q[58];
cz q[57],q[58];
u3(0.33347610999999994,-pi/2,pi/2) q[58];
cz q[57],q[58];
u3(pi/2,0,pi) q[58];
u3(pi/2,-1.6503585,-pi) q[59];
u3(pi/2,0.9489520732051036,-pi) q[60];
u3(1.8979041367948966,pi/2,-pi/2) q[61];
cz q[60],q[61];
u3(0.9489520699999999,-pi/2,pi/2) q[61];
cz q[60],q[61];
u3(pi/2,0,0.15912435358979327) q[60];
cz q[59],q[60];
u3(1.6503585000000003,pi/2,-pi/2) q[60];
cz q[59],q[60];
u3(pi/2,0,pi) q[60];
u3(pi/2,-1.3871256,-pi) q[61];
u3(pi/2,0.5998635732051039,-pi) q[62];
u3(1.1997272167948967,pi/2,-pi/2) q[63];
cz q[62],q[63];
u3(0.5998636099999998,-pi/2,pi/2) q[63];
cz q[62],q[63];
u3(pi/2,0,-0.367341453589793) q[62];
cz q[61],q[62];
u3(1.3871256000000003,pi/2,-pi/2) q[62];
cz q[61],q[62];
u3(pi/2,0,pi) q[62];
u3(pi/2,1.7412821000000003,-pi) q[63];
u3(pi/2,-0.19204912679489627,-pi) q[64];
u3(0.3840982732051033,-pi/2,pi/2) q[65];
cz q[64],q[65];
u3(0.19204913000000012,pi/2,-pi/2) q[65];
cz q[64],q[65];
u3(pi/2,0,-0.34097155358979325) q[64];
cz q[63],q[64];
u3(1.7412820999999998,-pi/2,pi/2) q[64];
cz q[63],q[64];
u3(pi/2,0,pi) q[64];
u3(pi/2,1.3090212999999995,-pi) q[65];
u3(pi/2,0.6566178732051036,-pi) q[66];
u3(1.3132357367948968,pi/2,-pi/2) q[67];
cz q[66],q[67];
u3(0.6566178699999998,-pi/2,pi/2) q[67];
cz q[66],q[67];
u3(pi/2,0,0.5235500535897932) q[66];
cz q[65],q[66];
u3(1.3090213,-pi/2,pi/2) q[66];
cz q[65],q[66];
u3(pi/2,0,pi) q[66];
u3(pi/2,-0.2587131899999999,-pi) q[67];
u3(pi/2,1.6385176803846901,-pi) q[68];
u3(3.0061498803846898,-pi/2,pi/2) q[69];
cz q[68],q[69];
u3(1.6385176999999997,-pi/2,pi/2) q[69];
cz q[68],q[69];
u3(pi/2,0,-2.6241662735897933) q[68];
cz q[67],q[68];
u3(0.2587131900000001,pi/2,-pi/2) q[68];
cz q[67],q[68];
u3(pi/2,0,pi) q[68];
u3(pi/2,-0.80006361,-pi) q[69];
u3(pi/2,-1.7755500967948965,-pi) q[70];
u3(2.7320851267948965,pi/2,-pi/2) q[71];
cz q[70],q[71];
u3(1.7755501000000002,pi/2,-pi/2) q[71];
cz q[70],q[71];
u3(pi/2,0,-1.5414654535897934) q[70];
cz q[69],q[70];
u3(0.80006361,pi/2,-pi/2) q[70];
cz q[69],q[70];
u3(pi/2,0,pi) q[70];
u3(pi/2,-0.11523598999999995,-pi) q[71];
u3(pi/2,0.17121037320510313,-pi) q[72];
u3(0.3424206267948967,pi/2,-pi/2) q[73];
cz q[72],q[73];
u3(0.17121032999999986,-pi/2,pi/2) q[73];
cz q[72],q[73];
u3(pi/2,0,-2.911120673589793) q[72];
cz q[71],q[72];
u3(0.11523599000000012,pi/2,-pi/2) q[72];
cz q[71],q[72];
u3(pi/2,0,pi) q[72];
u3(pi/2,0.17468742000000015,-pi) q[73];
u3(pi/2,0.42230427320510344,-pi) q[74];
u3(0.8446085567948965,pi/2,-pi/2) q[75];
cz q[74],q[75];
u3(0.4223042799999998,-pi/2,pi/2) q[75];
cz q[74],q[75];
u3(pi/2,0,2.7922178135897937) q[74];
cz q[73],q[74];
u3(0.1746874199999999,-pi/2,pi/2) q[74];
cz q[73],q[74];
u3(pi/2,0,pi) q[74];
u3(pi/2,1.0842291,-pi) q[75];
u3(pi/2,1.83468688038469,-pi) q[76];
u3(2.6138114803846895,-pi/2,pi/2) q[77];
cz q[76],q[77];
u3(1.8346869,-pi/2,pi/2) q[77];
cz q[76],q[77];
u3(pi/2,0,0.9731344535897932) q[76];
cz q[75],q[76];
u3(1.0842290999999997,-pi/2,pi/2) q[76];
cz q[75],q[76];
u3(pi/2,0,pi) q[76];
u3(pi/2,-1.6278165,-pi) q[77];
u3(pi/2,0.4499325732051034,-pi) q[78];
u3(0.8998651967948966,pi/2,-pi/2) q[79];
cz q[78],q[79];
u3(0.4499325999999999,-pi/2,pi/2) q[79];
cz q[78],q[79];
u3(pi/2,0,0.11404035358979314) q[78];
cz q[77],q[78];
u3(1.6278165,pi/2,-pi/2) q[78];
cz q[77],q[78];
u3(pi/2,0,pi) q[78];
u3(pi/2,1.7280132999999998,-pi) q[79];
u3(pi/2,1.5614570732051032,-pi) q[80];
u3(3.122914226794897,pi/2,-pi/2) q[81];
cz q[80],q[81];
u3(1.5614570999999995,-pi/2,pi/2) q[81];
cz q[80],q[81];
u3(pi/2,0,-0.3144339535897931) q[80];
cz q[79],q[80];
u3(1.7280132999999998,-pi/2,pi/2) q[80];
cz q[79],q[80];
u3(pi/2,0,pi) q[80];
u3(pi/2,-1.1939662,-pi) q[81];
u3(pi/2,1.5218395732051029,-pi) q[82];
u3(3.0436792267948967,pi/2,-pi/2) q[83];
cz q[82],q[83];
u3(1.5218395999999998,-pi/2,pi/2) q[83];
cz q[82],q[83];
u3(pi/2,0,-0.7536602535897936) q[82];
cz q[81],q[82];
u3(1.1939662000000002,pi/2,-pi/2) q[82];
cz q[81],q[82];
u3(pi/2,0,pi) q[82];
u3(pi/2,1.4127416000000004,-pi) q[83];
u3(pi/2,1.498717573205104,-pi) q[84];
u3(2.9974352267948965,pi/2,-pi/2) q[85];
cz q[84],q[85];
u3(1.4987175999999998,-pi/2,pi/2) q[85];
cz q[84],q[85];
u3(pi/2,0,0.31610945358979325) q[84];
cz q[83],q[84];
u3(1.4127415999999997,-pi/2,pi/2) q[84];
cz q[83],q[84];
u3(pi/2,0,pi) q[84];
u3(pi/2,0.5670681200000001,-pi) q[85];
u3(pi/2,-0.5912861367948965,-pi) q[86];
u3(1.182572273205103,-pi/2,pi/2) q[87];
cz q[86],q[87];
u3(0.5912861400000001,pi/2,-pi/2) q[87];
cz q[86],q[87];
u3(pi/2,0,2.0074564535897927) q[86];
cz q[85],q[86];
u3(0.5670681199999998,-pi/2,pi/2) q[86];
cz q[85],q[86];
u3(pi/2,0,pi) q[86];
u3(pi/2,1.1411537999999997,-pi) q[87];
u3(pi/2,-1.3634349967948967,-pi) q[88];
u3(2.7268699803846896,-pi/2,pi/2) q[89];
cz q[88],q[89];
u3(1.3634350000000002,pi/2,-pi/2) q[89];
cz q[88],q[89];
u3(pi/2,0,0.8592850535897929) q[88];
cz q[87],q[88];
u3(1.1411538,-pi/2,pi/2) q[88];
cz q[87],q[88];
u3(pi/2,0,pi) q[88];
u3(pi/2,-1.2546659,-pi) q[89];
u3(pi/2,-1.6723290967948965,-pi) q[90];
u3(2.9385271267948965,pi/2,-pi/2) q[91];
cz q[90],q[91];
u3(1.6723291000000002,pi/2,-pi/2) q[91];
cz q[90],q[91];
u3(pi/2,0,-0.6322608535897931) q[90];
cz q[89],q[90];
u3(1.2546659000000002,pi/2,-pi/2) q[90];
cz q[89],q[90];
u3(pi/2,0,pi) q[90];
u3(pi/2,0.6183749600000001,-pi) q[91];
u3(pi/2,-0.7927113567948965,-pi) q[92];
u3(1.5854226803846896,-pi/2,pi/2) q[93];
cz q[92],q[93];
u3(0.7927113600000002,pi/2,-pi/2) q[93];
cz q[92],q[93];
u3(pi/2,0,1.9048427535897936) q[92];
cz q[91],q[92];
u3(0.6183749599999999,-pi/2,pi/2) q[92];
cz q[91],q[92];
u3(pi/2,0,pi) q[92];
u3(pi/2,-0.4723730399999999,-pi) q[93];
u3(pi/2,-0.7831571767948966,-pi) q[94];
u3(1.5663143732051032,-pi/2,pi/2) q[95];
cz q[94],q[95];
u3(0.7831571800000001,pi/2,-pi/2) q[95];
cz q[94],q[95];
u3(pi/2,0,-2.1968465735897933) q[94];
cz q[93],q[94];
u3(0.47237304,pi/2,-pi/2) q[94];
cz q[93],q[94];
u3(pi/2,0,pi) q[94];
u3(pi/2,-0.8858608399999999,-pi) q[95];
u3(pi/2,-0.35141842679489654,-pi) q[96];
u3(0.7028368732051032,-pi/2,pi/2) q[97];
cz q[96],q[97];
u3(0.3514184200000001,pi/2,-pi/2) q[97];
cz q[96],q[97];
u3(pi/2,0,-1.3698709535897933) q[96];
cz q[95],q[96];
u3(0.88586084,pi/2,-pi/2) q[96];
cz q[95],q[96];
u3(pi/2,0,pi) q[96];
u3(pi/2,0,pi) q[97];
