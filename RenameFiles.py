import os

def rename_images(folder_path, prefix="image"):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
    
    for i, filename in enumerate(files, start=1):
        new_name = f"{prefix}{i}.jpg"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        
        # Dosya zaten varsa atla
        if os.path.exists(dst):
            print(f"SKIPPED: {dst} already exists.")
            continue
        
        os.rename(src, dst)
        print(f"{filename} -> {new_name}")

rename_images(r"C:\Users\HP\Desktop\MYZ307-Project\Caravaggio", prefix="caravaggio_image")
rename_images(r"C:\Users\HP\Desktop\MYZ307-Project\Alexandre Cabanel", prefix="cabanel_image")
rename_images(r"C:\Users\HP\Desktop\MYZ307-Project\Monet", prefix="monet_image")

        
