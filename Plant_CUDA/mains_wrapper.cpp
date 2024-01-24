#include <iostream>
#include <stdexcept>

void true_main();
void false_main();
void main_third();

int main()
	{
	auto function{&main_third};

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