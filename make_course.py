import os
import json

def create_directory_structure(course_structure):
    os.makedirs(course_structure["title"], exist_ok=True)
    os.chdir(course_structure["title"])

    for section_index, section in enumerate(course_structure["sections"]):
        section_directory = f"{section_index+1}-{section['title'].replace(' ', '_')}"
        os.makedirs(section_directory, exist_ok=True)
        os.chdir(section_directory)

        for subsection_index, subsection in enumerate(section["subsections"]):
            subsection_directory = f"{subsection_index+1}-{subsection['title'].replace(' ', '_')}"
            os.makedirs(subsection_directory, exist_ok=True)
            os.chdir(subsection_directory)

            with open("README.md", "w") as f:
                f.write(f"# {subsection['title']}\n\n")
                f.write(subsection["content"])

                if "exercise" in subsection:
                    f.write("\n\n## Exercise\n\n")
                    f.write(subsection["exercise"])

                if "project" in subsection:
                    f.write("\n\n## Project\n\n")
                    f.write(subsection["project"])

                if "reflection" in subsection:
                    f.write("\n\n## Reflection\n\n")
                    f.write(subsection["reflection"])

                if "quiz" in subsection:
                    f.write("\n\n## Quiz\n\n")
                    f.write(subsection["quiz"])

            os.chdir("..")

        os.chdir("..")

    os.chdir("..")


if __name__ == "__main__":
    with open("course_structure.json", "r") as f:
        course_structure = json.load(f)

    create_directory_structure(course_structure)
