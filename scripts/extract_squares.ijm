// Values to change if necessary

waitForUser("Please select the directory, that contains the subfolders for all patients");
par_dir = getDirectory("Please select the directory, that contains the subfolders for all patients");

//make output directory
out_dir = par_dir+"/extracts"
File.makeDirectory(out_dir);

// Extract Region from slice in coronal plane (Z-axis from front to back)

//specify the name of the extracted region
extract= "hippocampus";
extract_dir = out_dir+"/"+extract;
File.makeDirectory(extract_dir);

for (i=1; i<=457; i++){
	
	//convert i into 4-digit format
	long_i = "000"+i;
	long_i = substring(long_i, lengthOf(long_i)-4, lengthOf(long_i));
		
	//specify paths for images with either 3 or 4 replicates
	image_path_3 = par_dir+"/OAS1_"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_"+long_i+"_MR1_mpr_n"+3+"_anon_111_t88_gfc_cor_110.gif";
	image_path_4 = par_dir+"/OAS1_"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_"+long_i+"_MR1_mpr_n"+4+"_anon_111_t88_gfc_cor_110.gif";
	
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
		print("Image "+long_i+" does not exist");
		continue;
	}
		

	
	//draw rectangle and crop
	makeRectangle(45, 104, 86, 51);
	run("Crop");
	
	//save new image in output directory
	saveAs("Gif", extract_dir+"/OAS1_"+long_i+"_"+extract+".gif");
	close();
	
}


// Extract Region from slice in transverse plane (Z-axis from top to bottom)

//specify the type of extract
extract= "ventricles";
extract_dir = out_dir+"/"+extract;
File.makeDirectory(extract_dir);

for (i=1; i<=457; i++){
	
	//convert i into 4-digit format
	long_i = "000"+i;
	long_i = substring(long_i, lengthOf(long_i)-4, lengthOf(long_i));
		
	//specify paths for images with either 3 or 4 replicates
	image_path_3 = par_dir+"/OAS1_"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_"+long_i+"_MR1_mpr_n"+3+"_anon_111_t88_gfc_tra_90.gif";
	image_path_4 = par_dir+"/OAS1_"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_"+long_i+"_MR1_mpr_n"+4+"_anon_111_t88_gfc_tra_90.gif";
	
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
		print("Image "+long_i+" does not exist");
		continue;
	}
		

	//draw rectangle and crop
	makeRectangle(45, 49, 86, 106);
	run("Crop");
	
	//save new image in output directory
	saveAs("Gif", extract_dir+"/OAS1_"+long_i+"_"+extract+".gif");
	close();
	
}


// Extract Region from slice in saggital plane (Z-axis from side to side)


//specify the type of extract
extract= "parietal";
extract_dir = out_dir+"/"+extract;
File.makeDirectory(extract_dir);

for (i=1; i<=457; i++){
	
	//convert i into 4-digit format
	long_i = "000"+i;
	long_i = substring(long_i, lengthOf(long_i)-4, lengthOf(long_i));
		
	//specify paths for images with either 3 or 4 replicates
	image_path_3 = par_dir+"/OAS1_"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_"+long_i+"_MR1_mpr_n"+3+"_anon_111_t88_gfc_sag_95.gif";
	image_path_4 = par_dir+"/OAS1_"+long_i+"_MR1/PROCESSED/MPRAGE/T88_111/OAS1_"+long_i+"_MR1_mpr_n"+4+"_anon_111_t88_gfc_sag_95.gif";
	
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
		print("Image "+long_i+" does not exist");
		continue;
	}
		

	//draw rectangle and crop
	makeRectangle(6, 16, 72, 73);
	run("Crop");
	
	//save new image in output directory
	saveAs("Gif", extract_dir+"/OAS1_"+long_i+"_"+extract+".gif");
	close();
	
}