class elsciRLInput:
    def __init__(self, description_lookup:dict=None):
        self.description_lookup = description_lookup
        # New: store descriptions provided so the user doesn't need to provide multiple times
        self.descriptions_stored:dict={}
        
    def user_input(self):
        instructions = []
        instruction_descriptions = []
        while True:
            instr = input("Please provide the current instruction... ([e/exit] to end path)")
            if (instr == "e")|(instr=="exit"):
                break
            
            if not self.description_lookup:
                if instr not in self.descriptions_stored:
                    description = input("Please provide a description of the instruction...")
                else:
                    print("Instruction description provided previously.")
                    description = self.descriptions_stored[instr]
            if description == "None":
                description = instr
            
            instructions.append(instr)
            instruction_descriptions.append(description)
            self.descriptions_stored[instr] = description


        return instructions, instruction_descriptions