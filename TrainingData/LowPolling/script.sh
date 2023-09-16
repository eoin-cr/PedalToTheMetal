x_acc,y_acc,z_acc,time
#!/bin/bash

# Define the text you want to add to the beginning of each file
line_to_add="x_acc,y_acc,z_acc,time"

# Specify the directory where your files are located
directory="."

# Use a for loop to iterate over the files in the directory
for file in "$directory"/*; do
    # Check if the item is a file (not a directory)
    if [ -f "$file" ]; then
        # Use the cat command to concatenate the line and the file's content
        # Then overwrite the file with the new content
        echo "$line_to_add" | cat - "$file" > "$file.new"
        mv "$file.new" "$file"
        echo "Added to $file"
    fi
done

