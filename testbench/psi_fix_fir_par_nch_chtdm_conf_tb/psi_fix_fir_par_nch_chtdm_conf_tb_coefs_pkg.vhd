------------------------------------------------------------------------------
--  Copyright (c) 2018 by Paul Scherrer Institute, Switzerland
--  All rights reserved.
------------------------------------------------------------------------------

------------------------------------------------------------------------------
-- Libraries
------------------------------------------------------------------------------
library ieee;
	use ieee.std_logic_1164.all;
	use ieee.numeric_std.all;
	
library work;
	use work.psi_common_array_pkg.all;
	
------------------------------------------------------------------------------
-- Package Declaration
------------------------------------------------------------------------------
package psi_fix_fir_par_nch_chtdm_conf_tb_coefs_pkg is

	constant Coefs : t_areal(0 to 47) := (
		0.00087738037109375,
		0.001190185546875,
		0.00115203857421875,
		0.00055694580078125,
		-0.00072479248046875,
		-0.0024871826171875,
		-0.003997802734375,
		-0.00415802001953125,
		-0.00202178955078125,
		0.002532958984375,
		0.00824737548828125,
		0.01255035400390625,
		0.0124053955078125,
		0.00576019287109375,
		-0.006988525390625,
		-0.02217864990234375,
		-0.03334808349609375,
		-0.03308868408203125,
		-0.01572418212890625,
		0.020050048828125,
		0.0697784423828125,
		0.12380218505859375,
		0.16971588134765625,
		0.19608306884765625,
		0.19608306884765625,
		0.16971588134765625,
		0.12380218505859375,
		0.0697784423828125,
		0.020050048828125,
		-0.01572418212890625,
		-0.03308868408203125,
		-0.03334808349609375,
		-0.02217864990234375,
		-0.006988525390625,
		0.00576019287109375,
		0.0124053955078125,
		0.01255035400390625,
		0.00824737548828125,
		0.002532958984375,
		-0.00202178955078125,
		-0.00415802001953125,
		-0.003997802734375,
		-0.0024871826171875,
		-0.00072479248046875,
		0.00055694580078125,
		0.00115203857421875,
		0.001190185546875,
		0.00087738037109375);

end package;

------------------------------------------------------------------------------
-- Package Body
------------------------------------------------------------------------------
package body psi_fix_fir_par_nch_chtdm_conf_tb_coefs_pkg is 

end psi_fix_fir_par_nch_chtdm_conf_tb_coefs_pkg;
