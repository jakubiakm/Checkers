//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated from a template.
//
//     Manual changes to this file may cause unexpected behavior in your application.
//     Manual changes to this file will be overwritten if the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Checkers.Data
{
    using System;
    using System.Collections.Generic;
    
    public partial class player_information
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2214:DoNotCallOverridableMethodsInConstructors")]
        public player_information()
        {
            this.games = new HashSet<game>();
            this.games1 = new HashSet<game>();
        }
    
        public int player_information_id { get; set; }
        public int player_id { get; set; }
        public int algorithm_id { get; set; }
        public int number_of_pieces { get; set; }
        public Nullable<int> tree_depth { get; set; }
        public Nullable<double> uct_parameter { get; set; }
        public Nullable<int> number_of_iterations { get; set; }
    
        public virtual algorithm algorithm { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<game> games { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<game> games1 { get; set; }
        public virtual player player { get; set; }
    }
}
