waitForUser("Please locate the 'extracts' folder");
par_dir = getDirectory("Please locate the 'extracts' folder");

//make output directory
out_dir = par_dir+"/montage"
File.makeDirectory(out_dir);

for (i=1; i<=5; i++){
	long_i = "000"+i;
	long_i = substring(long_i, lengthOf(long_i)-4, lengthOf(long_i));
	if (File.exists(par_dir+"/hippocampus/OAS1_"+long_i+"_hippocampus.gif")) {
		open(par_dir+"/hippocampus/OAS1_"+long_i+"_hippocampus.gif");
		run("Rotate 90 Degrees Right");
		open(par_dir+"/parietal/OAS1_"+long_i+"_parietal.gif");
		open(par_dir+"/ventricles/OAS1_"+long_i+"_ventricles.gif");
		run("Images to Stack", "name=Stack_"+long_i+" use");
		run("Make Montage...", "columns=3 rows=1 scale=1");
		saveAs("Gif", out_dir+"/OAS1_"+long_i+"_moontage.gif");
	}
	else{
		continue;	
	}

}