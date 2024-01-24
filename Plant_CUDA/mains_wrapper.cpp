#include <iostream>
#include <stdexcept>

void true_main();
void false_main();

int main()
	{
	auto function{&true_main};

	/*
	function();
	/*/
	try { function(); }
	catch (const std::exception& e)
		{
		std::cout << e.what() << std::endl;
		}
	/**/
	return 0;
	}