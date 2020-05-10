function ShowHideDiv() {
    var chkImage = document.getElementById("chkImage");
    var img = document.getElementById("img_input_method");
	var img_des = document.getElementById("img_des_input_method");
		
	if(chkImage.checked){  // Input is an image
		//img.style.display = "block";
        img.style.opacity = "1";
        img.style.height = "auto";

		//img_des.style.display = "none";
		img_des.style.transition = "opacity 2s";
		img_des.style.opacity = "0";
        img_des.style.height = "0";
        img_des.style.overflow = "hidden";
	}else{  // Input is an image vector
		//img.style.display = "none";
		img.style.transition = "opacity 2s";
		img.style.opacity = "0";
        img.style.height = "0";
        img.style.overflow = "hidden";

		//img_des.style.display = "block";
		img_des.style.opacity = "1";
        img_des.style.height = "auto";
	}
}