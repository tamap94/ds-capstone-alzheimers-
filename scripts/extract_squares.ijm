par_dir = getDirectory("Select your folder, which contains the subfolders for all patients");

//make output directory
out_dir = par_dir+"/extracts"
File.makeDirectory(out_dir);

for (i=1; i<=416; i++){
	
	//convert i into 4-digit format
	long_i = "000"+i;
	long_i = substring(long_i, lengthOf(long_i)-3, lengthOf(long_i));
		
	//specify paths for images with either 3 or 4 replicates
	image_path_3 = par_dir+"/OAS1_0"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0"+long_i+"_MR1_mpr_n"+3+"_anon_111_t88_gfc_cor_110.gif";
	image_path_4 = par_dir+"/OAS1_0"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0"+long_i+"_MR1_mpr_n"+4+"_anon_111_t88_gfc_cor_110.gif";
	
	//try to open image
	if(File.exists(image_path_4)){
		 open(image_path_4);
		 print(image_path_4);
	}
	else if(File.exists(image_path_3)){
		open(image_path_3);
		print(image_path_3);
	}
	else{
		print("Image 0"+long_i+" does not exist");
		continue;
	}
	
	
	//specify the type of extract
	extract= "Hippocampus";
	extract_dir = out_dir+"/"+extract;
	File.makeDirectory(extract_dir);
	
	//draw rectangle and crop
	makeRectangle(42, 103, 86, 55);
	run("Crop");
	
	//save new image in output directory
	saveAs("Gif", extract_dir+"/OAS1_000"+i+"_"+extract+".gif");
	close();
}
